#!/bin/bash

dataset=audioset
# ckpt=/hpc_stor03/sjtu_home/zhisheng.zheng/models/audiomae/finetuned.pth
ckpt=/hpc_stor03/sjtu_home/zhisheng.zheng/models/binaural_encoder/all/checkpoint-12.pth

audioset_label=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/class_whitelist_encoder.csv
audioset_train_json=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/unbalanced_no_missing.json
audioset_train_weight=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/distributed/unbalanced.csv
audioset_eval_json=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/eval_no_missing.json

reverb_type=binaural
reverb_train_json=/data/shared/zsz01/SpatialAudio/reverb/mp3d/train_reverberation.json
reverb_val_json=/data/shared/zsz01/SpatialAudio/reverb/mp3d/eval_reverberation.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m debugpy --listen 55555 --wait-for-client -m torch.distributed.launch \
--nproc_per_node=1 --use_env main_finetune_as.py \
--log_dir /hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/outputs/eval \
--output_dir /hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/outputs/eval \
--model vit_base_patch16 \
--dataset $dataset \
--audioset_train $audioset_train_json \
--audioset_eval $audioset_eval_json \
--label_csv $audioset_label \
--reverb_train $reverb_train_json \
--reverb_val $reverb_val_json \
--reverb_type $reverb_type \
--finetune $ckpt \
--batch_size 1 \
--num_workers 4 \
--eval \
--dist_eval \
--nb_classes 355 \
--audio_normalize \