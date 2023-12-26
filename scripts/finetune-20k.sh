#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

blr=1e-3
mask_t_prob=0.25
mask_f_prob=0.25

dataset=audioset
ckpt=/home/zhisheng/models/audiomae/pretrained.pth

audioset_label=/saltpool0/data/zhisheng/audioset/class_whitelist_encoder.csv
audioset_train_json=/saltpool0/data/zhisheng/audioset/balanced_no_missing.json
audioset_eval_json=/saltpool0/data/zhisheng/audioset/eval_no_missing.json

reverb_type=binaural
reverb_train_json=/home/zhisheng/data/SpatialAudio/reverb/mp3d/train_reverberation.json
reverb_val_json=/home/zhisheng/data/SpatialAudio/reverb/mp3d/eval_reverberation.json

output_dir=/home/zhisheng/AudioMAE-fusion/outputs/finetune-20k-real-DP
log_dir=/home/zhisheng/AudioMAE-fusion/outputs/finetune-20k-real-DP

# -m debugpy --listen 55555 --wait-for-client
python -m torch.distributed.launch \
    --nproc_per_node=2 --use_env main_finetune_as.py \
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
    --batch_size 64 \
    --num_workers 0 \
    --roll_mag_aug \
    --mixup 0.5 \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 0 \
    --epochs 60 \
    --warmup_epochs 4 \
    --mask_2d \
    --nb_classes 355 \
