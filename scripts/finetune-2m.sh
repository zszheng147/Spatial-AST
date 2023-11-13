#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

blr=2e-4
mask_t_prob=0.2
mask_f_prob=0.2

dataset=audioset
ckpt=/home/zhisheng/models/AudioMAE/finetuned.pth

audioset_label=/saltpool0/data/zhisheng/audioset/class_labels_indices.csv
audioset_train_json=/saltpool0/data/zhisheng/audioset/unbalanced_no_missing.json
audioset_train_weight=/saltpool0/data/zhisheng/audioset/weights/unbalanced_weights.csv
audioset_eval_json=/saltpool0/data/zhisheng/audioset/eval_no_missing.json

output_dir=/home/zhisheng/scratch/projects/AudioMAE-spatial/outputs/finetune-2m
log_dir=/home/zhisheng/scratch/projects/AudioMAE-spatial/outputs/finetune-2m

python -m debugpy --listen 55555 --wait-for-client -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main_finetune_as.py \
    --log_dir $log_dir \
	--output_dir $output_dir \
    --model vit_base_patch16 \
    --dataset $dataset \
    --audioset_train $audioset_train_json \
    --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --weight_csv $audioset_train_weight \
    --finetune $ckpt \
    --blr $blr \
    --dist_eval \
    --batch_size 4 \
    --roll_mag_aug \
    --mixup 0.0 \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 \
    --epochs 100 \
    --warmup_epochs 10 \
    --weight_sampler \
    --distributed_wrapper \
    --mask_2d
