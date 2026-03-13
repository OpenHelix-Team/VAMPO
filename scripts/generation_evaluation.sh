#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
# SVD_MODEL_PATH="output/svd/pretrained"  

SVD_MODEL_PATH="output/svd/train_2026-01-12T10-50-42"  
CLIP_MODEL_PATH="checkpoints/clip-vit-base-patch32"      
VAL_DATASET_DIR="data/vpp_latent/opensource_robotdata/calvin"  

python generation_evaluation.py \
    --eval \
    --config video_conf/val_calvin_svd.yaml \
    --video_model_path "$SVD_MODEL_PATH" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --val_dataset_dir "$VAL_DATASET_DIR" 
