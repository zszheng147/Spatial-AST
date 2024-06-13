#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

blr=1e-3
mask_t_prob=0.25
mask_f_prob=0.25

# Download from https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link
ckpt=/hpc_stor03/sjtu_home/zhisheng.zheng/models/audiomae/pretrained.pth

# Sound source
dataset=audioset
audio_path_root=/data/shared/AudioSet
audioset_label=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/metadata/class_labels_indices_subset.csv
audioset_train_json=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/metadata/balanced.json
audioset_train_weight=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/metadata/weights/balanced_weight.csv
audioset_eval_json=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/metadata/eval.json

# For reverberation data, please visit https://huggingface.co/datasets/zhisheng01/SpatialSounds/blob/main/mp3d_reverb.zip
reverb_type=$1 # or mono
reverb_path_root=/data/shared/zsz01/SpatialAudio/reverb/mp3d
reverb_train_json=/data/shared/zsz01/SpatialAudio/reverb/mp3d/train_reverberation.json
reverb_val_json=/data/shared/zsz01/SpatialAudio/reverb/mp3d/eval_reverberation.json

# logging path
output_dir=/hpc_stor03/sjtu_home/zhisheng.zheng/Spatial-AST/outputs/2m/binaural/test
log_dir=/hpc_stor03/sjtu_home/zhisheng.zheng/Spatial-AST/outputs/2m/binaural/test/log

mkdir -p $output_dir

python -m debugpy --listen 55555 --wait-for-client -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=54633 --use_env main_finetune.py \
    --log_dir $log_dir --output_dir $output_dir \
    --model build_AST --dataset $dataset --finetune $ckpt \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label --weight_csv $audioset_train_weight \
    --nb_classes 355 \
    --reverb_path_root $reverb_path_root --reverb_type $reverb_type \
    --reverb_train $reverb_train_json --reverb_val $reverb_val_json \
    --blr $blr --dist_eval --batch_size 64 --num_workers 4 \
    --roll_mag_aug --mixup 0.5 --audio_normalize \
    --mask_t_prob $mask_t_prob --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 --epochs 100 --warmup_epochs 10 --epoch_len 10000 \
    --weight_sampler --distributed_wrapper --mask_2d