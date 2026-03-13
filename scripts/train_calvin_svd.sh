#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
# 执行训练命令
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +'%Y%m%d_%H%M%S').log"


accelerate launch \
    --main_process_port 29506 \
    step1_train_svd.py \
    --config video_conf/train_calvin_svd_sft.yaml \
    pretrained_model_path=checkpoints/svd-robot-calvin-ft \
    #2>&1 | tee "$LOG_FILE"
