# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.stat import calculate_stats, concat_all_gather


def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        mtl_loss_fn: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None, log_writer=None,
        args=None
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)

        waveforms, reverbs = batch[0], batch[1]
        targets, spaital_targets = batch[2], batch[3]

        targets = targets.to(device, non_blocking=True)
        distance = spaital_targets['distance'].long().to(device, non_blocking=True)
        azimuth = spaital_targets['azimuth'].long().to(device, non_blocking=True)
        elevation = spaital_targets['elevation'].long().to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        outputs = model(waveforms, reverbs, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        
        logits = torch.sort(torch.softmax(outputs[0], dim=1), descending=True, dim=1)[1]
        valid_indices = []
        for i in range(logits.size(0)):
            nonzero_indices = (targets[i] > 0).nonzero(as_tuple=True)[0]
            is_valid = 1 if all(idx.item() in logits[i, :len(nonzero_indices)] for idx in nonzero_indices) else 0
            valid_indices.append(is_valid)
        valid_indices = torch.tensor(valid_indices).float().to(device)

        loss1 = criterion(outputs[0], targets)
        loss2 = valid_indices * F.cross_entropy(outputs[1], distance, reduction='none')
        loss3 = valid_indices * F.cross_entropy(outputs[2], azimuth, reduction='none')
        loss4 = valid_indices * F.cross_entropy(outputs[3], elevation, reduction='none')

        loss = mtl_loss_fn([loss1, loss2, loss3, loss4])
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(data_loader, model, device, dist_eval=False):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    vids = []

    all_distance_preds = []
    all_distances = []
    doa_dists = []
    for batch in metric_logger.log_every(data_loader, 300, header):

        waveforms, reverbs = batch[0], batch[1]
        target, spaital_targets = batch[2], batch[3]

        target = target.to(device, non_blocking=True)
        # compute output

        output = model(waveforms, reverbs)
        # remark: 
        # 1. use concat_all_gather and --dist_eval for faster eval by distributed load over gpus
        # 2. otherwise comment concat_all_gather and remove --dist_eval one every gpu
        if dist_eval:
            cls_output = concat_all_gather(output[0].detach())
            target = concat_all_gather(target)
        outputs.append(cls_output)
        targets.append(target)
        all_distances.append(spaital_targets['distance'].numpy())
        all_distance_preds.append(torch.argmax(output[1], dim=1).detach().cpu().numpy())
        
        az_pred = torch.argmax(output[2], dim=1).detach().cpu().numpy()
        ele_pred = torch.argmax(output[3], dim=1).detach().cpu().numpy()
        az_gt = spaital_targets['azimuth'].long().numpy()
        ele_gt = spaital_targets['elevation'].long().numpy()
        doa_dist = distance_between_spherical_coordinates_rad(az_gt, ele_gt, az_pred, ele_pred)

        doa_dists.append(doa_dist)

    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    vids = [j for sub in vids for j in sub]
    np.save('inf_output.npy', {'vids':vids, 'embs_527':outputs, 'targets':targets})
    stats = calculate_stats(outputs, targets)

    AP = [stat['AP'] for stat in stats]
    mAP = np.mean([stat['AP'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))

    all_distance_preds = np.concatenate(all_distance_preds)
    all_distances = np.concatenate(all_distances)
    doa_dists = np.concatenate(doa_dists)

    total_samples = len(all_distances)
    spatial_outputs = []

    distance_correct = np.sum([1 for truth, pred in zip(all_distances, all_distance_preds) if abs(truth - pred) <= 1])
    spatial_outputs.append(distance_correct)

    threshold = 20
    doa_angular_error = np.sum(doa_dists)
    doa_error = np.sum(doa_dists > threshold) # 
    spatial_outputs.append(doa_error)
    spatial_outputs.append(doa_angular_error)

    if dist_eval:        
        spatial_outputs = torch.tensor(spatial_outputs).to(device)
        torch.distributed.all_reduce(spatial_outputs, op=torch.distributed.ReduceOp.SUM)
        
        total_samples = torch.tensor(total_samples).to(device)
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
        
        spatial_outputs = spatial_outputs.cpu().numpy()
        total_samples = total_samples.cpu().numpy()

    return {
        "mAP": mAP, "AP": AP, 
        "distance_accuracy": spatial_outputs[0]/total_samples,
        "doa_error": spatial_outputs[1]/total_samples,
        "doa_angular_error": spatial_outputs[2]/total_samples
    }
# @torch.no_grad()
# def evaluate(data_loader, model, device, dist_eval=False):
#     criterion = torch.nn.BCEWithLogitsLoss()

#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()
#     sed_outputs = []
#     sed_targets = []
#     vids = []

#     all_distance_preds = []
#     all_distances = []

#     doa_dists = []
#     for batch in metric_logger.log_every(data_loader, 300, header):

#         waveforms, reverbs = batch[0], batch[1]
#         target, spaital_targets = batch[2], batch[3]
#         # compute output

#         output = model(waveforms, reverbs)
#         # remark: 
#         # 1. use concat_all_gather and --dist_eval for faster eval by distributed load over gpus
#         # 2. otherwise comment concat_all_gather and remove --dist_eval one every gpu
#         # if dist_eval:
#         #     sed_outputs = concat_all_gather(output[0])
#         #     sed_targets = concat_all_gather(target)
#         sed_outputs.append(output[0].detach().cpu().numpy())
#         sed_targets.append(target.cpu().numpy())

#         all_distances.append(spaital_targets['distance'].numpy())
#         all_distance_preds.append(torch.argmax(output[1], dim=1).detach().cpu().numpy())
        
#         az_pred = torch.argmax(output[2], dim=1).detach().cpu().numpy()
#         ele_pred = torch.argmax(output[3], dim=1).detach().cpu().numpy()
#         az_gt = spaital_targets['azimuth'].long().numpy()
#         ele_gt = spaital_targets['elevation'].long().numpy()
#         doa_dist = distance_between_spherical_coordinates_rad(az_gt, ele_gt, az_pred, ele_pred)

#         doa_dists.append(doa_dist)

#     sed_outputs = np.concatenate(sed_outputs)
#     sed_targets = np.concatenate(sed_targets)
#     stats = calculate_stats(sed_outputs, sed_targets)
#     AP = [stat['AP'] for stat in stats]
#     mAP = np.mean([stat['AP'] for stat in stats])

#     all_distance_preds = np.concatenate(all_distance_preds)
#     all_distances = np.concatenate(all_distances)
#     doa_dists = np.concatenate(doa_dists)
#     total_samples = len(all_distances)
#     threshold = 20

#     distance_correct = np.sum([1 for truth, pred in zip(all_distances, all_distance_preds) if abs(truth - pred) <= 1])
#     doa_angular_error = np.sum(doa_dists)
#     doa_error = np.sum(doa_dists > threshold) # 

#     outputs = [mAP, distance_correct, doa_angular_error, doa_error]

#     if dist_eval:        
#         outputs = torch.tensor(outputs).to(device)
#         torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.SUM)
        
#         total_samples = torch.tensor(total_samples).to(device)
#         torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
        
#         outputs = outputs.cpu().numpy()
#         total_samples = total_samples.cpu().numpy()

#     return outputs, total_samples


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    az1 = (az1 - 180) * np.pi / 180.
    az2 = (az2 - 180) * np.pi / 180.
    ele1 = (ele1 - 90) * np.pi / 180.
    ele2 = (ele2 - 90) * np.pi / 180.

    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist