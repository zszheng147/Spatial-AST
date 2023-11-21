#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6

blr=1e-3
mask_t_prob=0.2
mask_f_prob=0.2

dataset=audioset
ckpt=/mnt/lustre/sjtu/home/zsz01/models/audiomae/finetuned.pth

audioset_label=/mnt/lustre/sjtu/home/zsz01/data/audioset/class_labels_indices.csv
audioset_train_json=/mnt/lustre/sjtu/home/zsz01/data/audioset/balanced_no_missing.json
audioset_eval_json=/mnt/lustre/sjtu/home/zsz01/data/audioset/eval_no_missing.json

reverb_type=BINAURAL
reverb_train_json=/mnt/lustre/sjtu/home/zsz01/remote/reverb/train_reverberation.json
reverb_val_json=/mnt/lustre/sjtu/home/zsz01/remote/reverb/eval_reverberation.json

output_dir=/mnt/lustre/sjtu/home/zsz01/AudioMAE-spatial/outputs/finetune-20k
log_dir=/mnt/lustre/sjtu/home/zsz01/AudioMAE-spatial/outputs/finetune-20k

# -m debugpy --listen 55555 --wait-for-client
python -m torch.distributed.launch \
    --nproc_per_node=6 --use_env main_finetune_as.py \
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
    --batch_size 96 \
    --num_workers 8 \
    --roll_mag_aug \
    --mixup 0.0 \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 0 \
    --epochs 60 \
    --warmup_epochs 4 \
    --mask_2d
