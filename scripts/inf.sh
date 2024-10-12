#!/bin/bash

dataset=audioset
ckpt=/path/to/trained_ckpt # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth

# Sound source
dataset=audioset
audio_path_root=/path/to/AudioSet # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/AudioSet
audioset_label=/path/to/metadata/class_labels_indices_subset.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv
audioset_train_json=/path/to/metadata/balanced.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/balanced.json
audioset_train_weight=/path/to/metadata/weights/balanced_weight.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/weights/balanced_weight.csv
audioset_eval_json=/path/to/metadata/eval.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/eval.json

# For reverberation data, please visit https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_type=$1 # or mono
reverb_path_root=/path/to/mp3d_reverb # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_train_json=/path/to/mp3d_reverb/train_reverberation.json
reverb_val_json=/path/to/mp3d_reverb/mp3d/eval_reverberation.json

# logging path
output_dir=./outputs/eval
log_dir=./outputs/eval/log


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main_finetune.py \
    --log_dir ${log_dir} --output_dir ${output_dir} \
    --model build_AST --dataset $dataset --finetune $ckpt \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label --nb_classes 355 \
    --reverb_path_root $reverb_path_root --reverb_type $reverb_type \
    --reverb_train $reverb_train_json --reverb_val $reverb_val_json \
    --batch_size 64 --num_workers 4 \
    --audio_normalize \
    --eval --dist_eval