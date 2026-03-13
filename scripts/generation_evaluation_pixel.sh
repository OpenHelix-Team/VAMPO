#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SVD_MODEL_PATH="output/svd/train_2025-12-30T15-00-10"  
CLIP_MODEL_PATH="checkpoints/clip-vit-base-patch32"      
VAL_DATASET_DIR="data/vpp_latent/opensource_robotdata/calvin"  

python generation_evaluation_pixel.py \
    --eval \
    --config video_conf/val_calvin_svd.yaml \
    --video_model_path "$SVD_MODEL_PATH" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --val_dataset_dir "$VAL_DATASET_DIR" 
