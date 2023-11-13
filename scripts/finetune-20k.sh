#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

blr=1e-3
mask_t_prob=0.2
mask_f_prob=0.2

dataset=audioset
ckpt=/home/zhisheng/models/AudioMAE/finetuned.pth

audioset_label=/saltpool0/data/zhisheng/audioset/class_labels_indices.csv
audioset_train_json=/saltpool0/data/zhisheng/audioset/balanced_no_missing.json
audioset_eval_json=/saltpool0/data/zhisheng/audioset/eval_no_missing.json

reverb_type=BINAURAL
reverb_train_json=/home/zhisheng/data/SpatialSound/reverberation/mp3d/train_reverberation.json
reverb_val_json=/home/zhisheng/data/SpatialSound/reverberation/mp3d/eval_reverberation.json

output_dir=/home/zhisheng/scratch/projects/AudioMAE-spatial/outputs/finetune-20k
log_dir=/home/zhisheng/scratch/projects/AudioMAE-spatial/outputs/finetune-20k


python -m debugpy --listen 55555 --wait-for-client -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main_finetune_as.py \
	--log_dir $log_dir \
	--output_dir $output_dir \
    --model vit_base_patch16 \
    --dataset $dataset \
    --audioset_train $audioset_train_json \
    --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --reverb_train $reverb_train_json \
    --reverb_val $reverb_val_json \
    --reverb_type $reverb_type \
    --finetune $ckpt \
    --blr $blr \
    --dist_eval \
    --batch_size 8 \
    --roll_mag_aug \
    --mixup 0.0 \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 15 \
    --epochs 60 \
    --warmup_epochs 4 \
    --mask_2d
