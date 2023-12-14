#!/bin/bash

dataset=audioset
# ckpt=/mnt/lustre/sjtu/home/zsz01/models/audiomae/pretrained.pth
ckpt=/mnt/lustre/sjtu/home/zsz01/AudioMAE-fusion/outputs/finetune-2m-DP-conv3x3/best-bn-rename.pth

audioset_label=/mnt/lustre/sjtu/home/zsz01/data/audioset/class_labels_indices.csv
audioset_train_json=/mnt/lustre/sjtu/home/zsz01/data/audioset/unbalanced_no_missing.json
audioset_train_weight=/mnt/lustre/sjtu/home/zsz01/data/audioset/distributed/unbalanced.csv
audioset_eval_json=/mnt/lustre/sjtu/home/zsz01/data/audioset/eval_no_missing.json

reverb_type=BINAURAL
reverb_train_json=/mnt/lustre/sjtu/home/zsz01/remote/reverb/train_reverberation.json
reverb_val_json=/mnt/lustre/sjtu/home/zsz01/remote/reverb/eval_reverberation.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main_finetune_as.py \
--log_dir /mnt/lustre/sjtu/home/zsz01/AudioMAE-fusion/outputs/eval \
--output_dir /mnt/lustre/sjtu/home/zsz01/AudioMAE-fusion/outputs/eval \
--model vit_base_patch16 \
--dataset $dataset \
--audioset_train $audioset_train_json \
--audioset_eval $audioset_eval_json \
--label_csv $audioset_label \
--reverb_train $reverb_train_json \
--reverb_val $reverb_val_json \
--reverb_type $reverb_type \
--finetune $ckpt \
--batch_size 32 \
--num_workers 4 \
--eval \
--dist_eval \