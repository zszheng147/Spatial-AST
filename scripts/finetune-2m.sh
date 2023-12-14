#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

blr=2e-4
mask_t_prob=0.25
mask_f_prob=0.25

dataset=audioset
# ckpt=/mnt/lustre/sjtu/home/zsz01/models/audiomae/pretrained.pth
ckpt=/mnt/lustre/sjtu/home/zsz01/AudioMAE-fusion/outputs/finetune-2m-logmagn/checkpoint-60.pth

audioset_label=/mnt/lustre/sjtu/home/zsz01/data/audioset/class_labels_indices.csv
audioset_train_json=/mnt/lustre/sjtu/home/zsz01/data/audioset/unbalanced_no_missing.json
audioset_train_weight=/mnt/lustre/sjtu/home/zsz01/data/audioset/distributed/unbalanced.csv
audioset_eval_json=/mnt/lustre/sjtu/home/zsz01/data/audioset/eval_no_missing.json

reverb_type=BINAURAL
reverb_train_json=/mnt/lustre/sjtu/home/zsz01/remote/reverb/train_reverberation.json
reverb_val_json=/mnt/lustre/sjtu/home/zsz01/remote/reverb/eval_reverberation.json

output_dir=/mnt/lustre/sjtu/home/zsz01/AudioMAE-fusion/outputs/finetune-2m-logmagn-allloss
log_dir=/mnt/lustre/sjtu/home/zsz01/AudioMAE-fusion/outputs/finetune-2m-logmagn-allloss

# -m debugpy --listen 55555 --wait-for-client
python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=45352 --use_env main_finetune_as.py \
    --log_dir $log_dir \
	--output_dir $output_dir \
    --model vit_base_patch16 \
    --dataset $dataset \
    --audioset_train $audioset_train_json \
    --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --weight_csv $audioset_train_weight \
    --reverb_train $reverb_train_json \
    --reverb_val $reverb_val_json \
    --reverb_type $reverb_type \
    --finetune $ckpt \
    --blr $blr \
    --dist_eval \
    --batch_size 64 \
    --num_workers 4 \
    --roll_mag_aug \
    --mixup 0.5 \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 0 \
    --epochs 50 \
    --warmup_epochs 0 \
    --weight_sampler \
    --distributed_wrapper \
    --mask_2d \
