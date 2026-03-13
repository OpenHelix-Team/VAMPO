import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import json

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL, UNetSpatioTemporalConditionModel
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
# from diffusers.models.attention_processor import AttnProcessor2_0, Attention
# from diffusers.models.attention import BasicTransformerBlock
# from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from einops import rearrange, repeat
import imageio
from video_models.pipeline import MaskStableVideoDiffusionPipeline,TextStableVideoDiffusionPipeline
import wandb
from decord import VideoReader, cpu
import decord
import lpips
from torch.nn import MSELoss
from torchvision.models.optical_flow import raft_large

    
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# def encode_text(texts, tokenizer, text_encoder, position_encode=True):
#     with torch.no_grad():
#         inputs = tokenizer(texts, padding='max_length', return_tensors="pt",truncation=True, max_length=20).to(text_encoder.device)
#         outputs = text_encoder(**inputs)
#         encoder_hidden_states = outputs.last_hidden_state # (batch, 30, 512)

#         if position_encode:
#             embed_dim, pos_num = encoder_hidden_states.shape[-1], encoder_hidden_states.shape[1]
#             pos = np.arange(pos_num,dtype=np.float64)

#             position_encode = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
#             position_encode = torch.tensor(position_encode, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype, requires_grad=False)

#             # print("position_encode",position_encode.shape)
#             # print("encoder_hidden_states",encoder_hidden_states.shape)

#             encoder_hidden_states += position_encode

#         encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=-1)

#     return encoder_hidden_states

def encode_text(texts, tokenizer, text_encoder, img_cond=None, img_cond_mask=None, img_encoder=None, position_encode=True, use_clip=True, args=None):
    max_length = args.clip_token_length
    with torch.no_grad():
        if use_clip:
            inputs = tokenizer(texts, padding='max_length', return_tensors="pt",truncation=True, max_length=max_length).to(text_encoder.device)
            outputs = text_encoder(**inputs)
            encoder_hidden_states = outputs.last_hidden_state # (batch, 30, 512)
            if position_encode:
                embed_dim, pos_num = encoder_hidden_states.shape[-1], encoder_hidden_states.shape[1]
                pos = np.arange(pos_num,dtype=np.float64)

                position_encode = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
                position_encode = torch.tensor(position_encode, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype, requires_grad=False)

                # print("position_encode",position_encode.shape)
                # print("encoder_hidden_states",encoder_hidden_states.shape)

                encoder_hidden_states += position_encode
            assert encoder_hidden_states.shape[-1] == 512

            if img_encoder is not None:
                assert img_cond is not None
                assert img_cond_mask is not None
                # print("img_encoder",img_encoder.shape)
                img_cond = img_cond.to(img_encoder.device)
                if len(img_cond.shape) == 5:
                    img_cond = img_cond.squeeze(1)
                
                img_hidden_states = img_encoder(img_cond).image_embeds
                img_hidden_states[img_cond_mask] = 0.0
                img_hidden_states = img_hidden_states.unsqueeze(1).expand(-1,encoder_hidden_states.shape[1],-1)
                assert img_hidden_states.shape[-1] == 512
                encoder_hidden_states = torch.cat([encoder_hidden_states, img_hidden_states], dim=-1)
                assert encoder_hidden_states.shape[-1] == 1024
            else:
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=-1)
        
        else:
            inputs = tokenizer(texts, padding='max_length', return_tensors="pt",truncation=True, max_length=32).to(text_encoder.device)
            outputs = text_encoder(**inputs)
            encoder_hidden_states = outputs.last_hidden_state # (batch, 30, 512)
            assert encoder_hidden_states.shape[1:] == (32,1024)

    return encoder_hidden_states

def eval(pipeline, text, tokenizer, text_encoder, img_cond, img_cond_mask, img_encoder, true_video, args, lpips_loss_fn, raft_model):
    
    # --- 1. 定义批次大小 ---
    # 定义用于评估的批次大小。如果显存足够，可以设置为 args.batch_size；
    # 否则，请设置为一个较小的固定值，如 4 或 8。
    eval_batch_size = 16 # <--- 建议修改这里的数值来控制显存占用
    
    device = pipeline.device
    B, T, C, H, W = true_video.shape
    
    # --- 2. 文本编码 (Text Encoding) ---
    with torch.no_grad():
        print("position_encode", args.position_encode)
        # 假设 encode_text 可以处理批次输入
        text_token = encode_text(text, tokenizer, text_encoder, img_cond=img_cond, img_cond_mask=img_cond_mask, img_encoder=img_encoder, position_encode=args.position_encode, args=args)

    # 初始化存储生成的视频和所有得分
    all_generated_videos = []
    all_lpips_scores = []
    all_mse_scores = []
    all_raft_dists = []

    # --- 3. 核心循环：按批次进行视频生成和指标计算 ---
    for i in tqdm(range(0, B, eval_batch_size), desc="Evaluating Batches"):
        
        # 确定当前批次的索引范围
        current_batch_end = min(i + eval_batch_size, B)
        current_batch_slice = slice(i, current_batch_end)
        current_batch_size = current_batch_end - i
        
        # 获取当前批次的输入
        image_batch = true_video[current_batch_slice, 0].to(device) # (B_curr, H, W, 3)
        text_token_batch = text_token[current_batch_slice].to(device)
        true_video_batch = true_video[current_batch_slice].to(device) # (B_curr, T, C, H, W)

        # --- A. 视频生成 (Pipeline Call) ---
        with torch.no_grad():
            videos = MaskStableVideoDiffusionPipeline.__call__(
                pipeline,
                image=image_batch,
                text=text_token_batch,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None
            ).frames # frames 是一个包含 PIL 图像列表的列表
        
        # 将 PIL 图像转换为 numpy 数组和 PyTorch tensor
        batch_videos_np = []
        for id_video, video in enumerate(videos):
            video_np = [np.array(frame) for frame in video]
            batch_videos_np.append(video_np)
        
        # 转换为 (B_curr, T, C, H, W) 格式的 Tensor，并归一化到 [0, 1]
        generated_video_tensor = torch.from_numpy(np.array(batch_videos_np)).permute(0, 1, 4, 2, 3).float() / 255.0
        all_generated_videos.append(generated_video_tensor)

        # 归一化真实视频到 [0, 1]
        true_video_tensor_norm = (true_video_batch.float() + 1.0) / 2.0 

        # --- B. LPIPS 损失计算 (分块处理) ---
        # LPIPS 需要输入在 [-1, 1] 范围
        generated_for_lpips = (generated_video_tensor.reshape(current_batch_size * T, C, H, W) * 2.0 - 1.0).to(device)
        true_for_lpips = true_video_batch.reshape(current_batch_size * T, C, H, W)
        with torch.no_grad():
            # 为避免大批次导致的显存或计算问题，分块处理
            lpips_scores_batch = []
            chunk_size = 256  # 可以根据显存调整
            num_chunks = int(math.ceil(generated_for_lpips.shape[0] / chunk_size))
            for j in range(num_chunks):
                start_idx = j * chunk_size
                end_idx = min((j + 1) * chunk_size, generated_for_lpips.shape[0])
                
                gen_chunk = generated_for_lpips[start_idx:end_idx]
                true_chunk = true_for_lpips[start_idx:end_idx]
                
                lpips_val = lpips_loss_fn(gen_chunk, true_chunk)
                lpips_scores_batch.extend(lpips_val.flatten().tolist())
                
            lpips_score = np.mean(lpips_scores_batch)
            all_lpips_scores.append(lpips_score)

        # --- C. MSE 损失计算 ---
        mse_loss_fn = MSELoss()
        with torch.no_grad():
            mse_score = mse_loss_fn(generated_video_tensor.to(device), true_video_tensor_norm).item()
            all_mse_scores.append(mse_score)

        # --- D. RAFT/光流损失计算 (分块处理) ---
        flow_dist_per_item = []
        generated_video_raft = (generated_video_tensor.clamp(0, 1) * 255).float().to(device)
        true_video_raft = (true_video_tensor_norm.clamp(0, 1) * 255).float().to(device)

        chunk_size_raft = 64  # 为RAFT设置一个较小的块大小
        num_chunks_raft = int(math.ceil(current_batch_size / chunk_size_raft))

        for j in range(T - 1):
            flow_dists_for_frame_j = []
            with torch.no_grad():
                for k in range(num_chunks_raft):
                    start_idx = k * chunk_size_raft
                    end_idx = min((k + 1) * chunk_size_raft, current_batch_size)

                    true_video_raft_chunk_t0 = true_video_raft[start_idx:end_idx, j]
                    true_video_raft_chunk_t1 = true_video_raft[start_idx:end_idx, j+1]
                    generated_video_raft_chunk_t0 = generated_video_raft[start_idx:end_idx, j]
                    generated_video_raft_chunk_t1 = generated_video_raft[start_idx:end_idx, j+1]

                    true_flow = raft_model(true_video_raft_chunk_t0, true_video_raft_chunk_t1)[-1]
                    pred_flow = raft_model(generated_video_raft_chunk_t0, generated_video_raft_chunk_t1)[-1]
                    
                    flow_dist_chunk = F.mse_loss(pred_flow, true_flow, reduction='none').mean(dim=[1, 2, 3])
                    flow_dists_for_frame_j.append(flow_dist_chunk)
            
            flow_dist_per_item.append(torch.cat(flow_dists_for_frame_j))

        # 将当前批次的平均光流距离存储起来
        raft_dist_batch_mean = torch.mean(torch.stack(flow_dist_per_item)).item()
        all_raft_dists.append(raft_dist_batch_mean)


    # --- 4. 汇总所有批次的平均得分 ---
    
    # 注意：这里的平均需要根据每个批次的大小进行加权平均，
    # 但由于您的原始代码使用了简单的平均，我们保持简单平均，假设 eval_batch_size 是 B 的因子，或者您不需要精确加权。
    
    final_mse_score = np.mean(all_mse_scores)
    final_lpips_score = np.mean(all_lpips_scores)
    final_raft_score = np.mean(all_raft_dists)

    return final_mse_score, final_lpips_score, final_raft_score

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, pipeline.vae, pipeline.unet

def main_eval(
    pretrained_model_path,
    clip_model_path,
    args,
    texts,
    img_conds,
    img_cond_masks,
    true_videos,
):

    # Load scheduler, tokenizer and models.
    pipeline, _, _ = load_primary_models(pretrained_model_path, eval=True)
    device = torch.device("cuda")
    pipeline.to(device)
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    # RAFT calculation
    raft_model = raft_large(progress=False).to(device)
    raft_model.eval()
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path,use_fast=False)
    text_encoder.requires_grad_(False).to(device)

    mse_score, lpips_score, raft_score = eval(pipeline, texts, tokenizer, text_encoder, img_conds, img_cond_masks, image_encoder, true_videos, args, lpips_loss_fn, raft_model)
    return mse_score, lpips_score, raft_score
    # eval(pipeline, tokenizer, text_encoder, true_videos, texts, args, pretrained_model_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="video_conf/val_svd.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--video_model_path", type=str, default="output/svd")
    parser.add_argument("--clip_model_path", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--val_dataset_dir", type=str, default='video_dataset_instance/bridge')
    parser.add_argument("--val_idx", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    val_args = args_dict.validation_args
    val_args.val_dataset_dir = args.val_dataset_dir

    if args.val_idx is not None:
        idxs = args.val_idx.split(",")
        idxs = [int(idx) for idx in idxs]
        val_args.val_idx = idxs

    from video_dataset.video_transforms import Resize_Preprocess, ToTensorVideo
    preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess((256,256)), # 288 512
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    image_encoder = None
    if val_args.use_img_cond:
        # load image encoder
        from transformers import CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
        image_encoder.requires_grad_(False)
        image_encoder.to(device)
        
        preprocess_clip = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple([args.clip_img_size, args.clip_img_size])), # 224,224
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258,0.27577711], inplace=True)
        ])
        print("use image condition")

    
    true_videos = []
    texts = []
    img_conds = []
    img_cond_masks = []

    input_dir = args.val_dataset_dir
    id_list = [i for i in range(128)]

    for id in tqdm(id_list, desc="Processing Annotations"):        # prepare original instruction    
        annotation_path = f"{input_dir}/annotation/val_d/{id}.json"
        # annotation_path = f"{input_dir}/annotation/train/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
            text_id = anno['texts'][0]
            # you can use new instruction to replace the original instruction in val_svd.yaml
            if val_args.use_new_instru:
                text_id = args.new_instru
            # text_id = "Put the green block above the blue block."
            texts.append(text_id)

        # prepare ground-truth video
        video_path = anno['videos'][val_args.camera_idx]['video_path']
        video_path = f"{input_dir}/{video_path}"
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        try:
            true_video = vr.get_batch(range(length)).asnumpy()
        except:
            true_video = vr.get_batch(range(length)).numpy()
        true_video = true_video.astype(np.uint8)
        true_video = torch.from_numpy(true_video).permute(0, 3, 1, 2) # (l, c, h, w)
        if not val_args.only_one_clip:
            skip = val_args.skip_step
            start_idx = val_args.start_idx
            end_idx = start_idx + int(val_args.num_frames*skip)
            if true_video.size(0) < end_idx:
                true_video = torch.concat([true_video, true_video[-1].unsqueeze(0).repeat(end_idx-true_video.size(0),1,1,1)], dim=0)
            true_video = true_video[start_idx:end_idx]
            true_video = true_video[::skip]
            # print("true_video",true_video.size(), start_idx, end_idx)
        else:
            idx = np.linspace(0,length-1,16).astype(int)
            true_video = true_video[idx]
        true_video = preprocess(true_video).unsqueeze(0)
        true_videos.append(true_video)
        
        if val_args.use_img_cond:
            # prepare image condition
            img_cond_masks.append(False if 'xhand' in video_path else True)
            cond_cam_idx = 1 if 'xhand' in video_path else 0
            # video_path_cond = anno['videos'][cond_cam_idx]['image_path']
            video_path_cond = anno['videos'][args.camera_idx]['video_path']
            video_path_cond = f"{input_dir}/{video_path_cond}"
            vr = decord.VideoReader(video_path_cond)
            frames = vr[start_idx].asnumpy()
            frames = torch.from_numpy(frames).permute(2,0,1).unsqueeze(0)
            frames = preprocess_clip(frames)

            img_conds.append(frames)

    true_videos = torch.cat(true_videos, dim=0)
    img_conds = torch.cat(img_conds, dim=0) if val_args.use_img_cond else None
    img_cond_masks = torch.tensor(img_cond_masks).to(device) if val_args.use_img_cond else None
    print("true_video size:",true_videos.size())
    print("instructions lenght:", len(texts))
    print("image condition mask:", img_cond_masks)

    if os.path.isdir(args.video_model_path):
        checkpoint_paths = [os.path.join(args.video_model_path, d) for d in os.listdir(args.video_model_path) if d.startswith('checkpoint-')]
        checkpoint_paths.sort(key=lambda x: int(x.split('-')[-1]))
        output_dir = args.video_model_path
    else:
        checkpoint_paths = [args.video_model_path]
        output_dir = os.path.dirname(args.video_model_path)

    results = {}
    for ckpt_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {ckpt_path}")
        mse, lpips_score, raft_score = main_eval(pretrained_model_path=ckpt_path,
                                     clip_model_path=args.clip_model_path,
                                     args=val_args,
                                     texts=texts,
                                     img_conds=img_conds,
                                     img_cond_masks=img_cond_masks,
                                     true_videos=true_videos
                                    )
        results[os.path.basename(ckpt_path)] = {"mse": mse, "lpips": lpips_score, "raft": raft_score}
        print("mse:", mse, "lpips:", lpips_score, "raft:", raft_score)
    results_path = os.path.join(output_dir, "pixel_evaluation_results_20steps.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {results_path}")

# bridge
# python make_prediction.py --eval --config video_conf/val_svd.yaml --video_model_path /cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-03-28T20-25-31/checkpoint-240000 --clip_model_path /cephfs/shared/llm/clip-vit-base-patch32 --val_dataset_dir video_dataset_instance/bridge --val_idx 2+10+8+14

# rt1
# python make_prediction.py --eval --config video_conf/val_svd.yaml --video_model_path /cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-03-28T20-25-31/checkpoint-240000 --clip_model_path /cephfs/shared/llm/clip-vit-base-patch32 --val_dataset_dir video_dataset_instance/rt1 --val_idx 1+6+8+10

# sthv2
# python make_prediction.py --eval --config video_conf/val_svd.yaml --video_model_path /cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-04-24T13-02-34/checkpoint-260000 --clip_model_path /cephfs/shared/llm/clip-vit-base-patch32 --val_dataset_dir video_dataset_instance/xhand  --val_idx 0+50+100+150
