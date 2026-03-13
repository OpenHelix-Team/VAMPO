#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

VIDEO_MODEL_PATH="checkpoints/svd-robot-calvin-ft"
ACTION_MODEL_FOLDER="checkpoints/dp-calvin"
CLIP_MODEL_PATH="checkpoints/clip-vit-base-patch32"
CALVIN_ABC_DIR="data/calvin_abc/task_ABC_D"

python policy_evaluation/calvin_evaluate.py \
    --video_model_path "$VIDEO_MODEL_PATH" \
    --action_model_folder "$ACTION_MODEL_FOLDER" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --calvin_abc_dir "$CALVIN_ABC_DIR"
