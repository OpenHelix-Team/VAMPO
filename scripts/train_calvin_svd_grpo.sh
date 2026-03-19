#!/bin/bash
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +'%Y%m%d_%H%M%S')_latent_lr1e-6.log"
  
accelerate launch \
    --main_process_port 29506 \
    step1_train_svd_grpo.py \
    --config video_conf/train_calvin_svd.yaml \
    pretrained_model_path=checkpoints/svd-robot-calvin-ft \
    sample_batch_size=8 \
    train_batch_size=8 \
    learning_rate=1e-6 \
    max_train_steps=10000 \
    checkpointing_steps=50 \
    validation_steps=50 \
    train_args.use_lora=False \
    train_args.lora_rank=16 \
    train_args.num_generations=8 \
    train_args.num_inner_epochs=1 \
    train_args.num_inference_steps=20 \
    train_args.guidance_scale=7.5 \
    train_args.clip_range=0.2 \
    train_args.adv_clip_max=2.0 \
    train_args.max_grad_norm=0.5 \
    train_args.use_weight=False \
    2>&1 | tee "$LOG_FILE"