#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

accelerate launch step2_train_action_calvin.py \
    --root_data_dir calvin_abc/task_ABC_D \
    --video_model_path output/svd/train_2026-01-12T10-50-42/checkpoint-200  \
    --text_encoder_path checkpoints/clip-vit-base-patch32
