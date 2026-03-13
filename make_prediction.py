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

# def eval(pipeline, text, tokenizer, text_encoder, img_cond, img_cond_mask, img_encoder, true_video, args, pretrained_model_path):
#     mask_frame_num = 1
#     image = true_video[:,0] # (batch, 256, 256, 3)
#     with torch.no_grad():
#         print("position_encode",args.position_encode)
#         text_token = encode_text(text, tokenizer, text_encoder, img_cond=img_cond, img_cond_mask=img_cond_mask, img_encoder=img_encoder, position_encode=args.position_encode, args=args)
#     # import pdb; pdb.set_trace()
#     videos = MaskStableVideoDiffusionPipeline.__call__(
#         pipeline,
#         image=image,
#         text=text_token,
#         width=args.width,
#         height=args.height,
#         num_frames=args.num_frames,
#         num_inference_steps=args.num_inference_steps,
#         decode_chunk_size=args.decode_chunk_size,
#         fps=args.fps,
#         motion_bucket_id=args.motion_bucket_id,
#         mask=None
#     ).frames
    
#     # import pdb; pdb.set_trace()
#     true_video = true_video.detach().cpu().numpy().transpose(0,1,3,4,2) #(2,16,256,256,3)
#     true_video = ((true_video+1)/2*255).astype(np.uint8)

#     new_videos = []
#     for id_video, video in enumerate(videos):
#         new_video = []
#         for idx, frame in enumerate(video):
#             new_video.append(np.array(frame))
#             # print("frame",frame)
#         new_videos.append(new_video)
#     videos = new_videos

#     videos = np.array([np.array(video) for video in videos]) #(2,16,256,256,3)
#     # videos = np.concatenate([true_video[:,:mask_frame_num],videos[:,mask_frame_num:]],axis=1)
#     # print("true_video",true_video.shape,"videos",videos.shape)
#     frame_num = videos.shape[1]
#     videos = np.concatenate([true_video[:,:frame_num],videos],axis=-3) #(2,16,512,256,3)
#     videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)
    
#     time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     day = datetime.datetime.now().strftime("%Y-%m-%d")

#     dataset_name = args.val_dataset_dir.split("/")[-1]
#     output_dir = f"{args.output_path}/{dataset_name}"
#     os.makedirs(output_dir, exist_ok=True)
#     model_name = pretrained_model_path.split("/")[-2]
#     model_id = pretrained_model_path.split("/")[-1]
#     filename = f"{output_dir}/{time}_{model_name}_{model_id}_{args.start_idx}.mp4"
#     writer = imageio.get_writer(filename, fps=4)
#     for frame in videos:
#         writer.append_data(frame)
#     writer.close()
#     return 

def eval(pipeline, text, tokenizer, text_encoder, img_cond, img_cond_mask, img_encoder, true_video, args, pretrained_model_path):
    mask_frame_num = 1
    image = true_video[:, 0]  # (batch, 256, 256, 3)

    with torch.no_grad():
        print("position_encode", args.position_encode)
        text_token = encode_text(
            text, tokenizer, text_encoder,
            img_cond=img_cond, img_cond_mask=img_cond_mask,
            img_encoder=img_encoder, position_encode=args.position_encode, args=args
        )

    # --- Generate video using pipeline ---
    videos = MaskStableVideoDiffusionPipeline.__call__(
        pipeline,
        image=image,
        text=text_token,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.decode_chunk_size,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        mask=None
    ).frames

    # --- Prepare true video ---
    true_video = true_video.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
    true_video = ((true_video + 1) / 2 * 255).astype(np.uint8)

    # --- Convert generated frames to numpy ---
    new_videos = []
    for id_video, video in enumerate(videos):
        new_video = []
        for idx, frame in enumerate(video):
            new_video.append(np.array(frame))
        new_videos.append(new_video)

    videos = np.array([np.array(video) for video in new_videos])  # (B, F, H, W, 3)

    # --- Concatenate true_video and generated video vertically ---
    frame_num = videos.shape[1]
    videos = np.concatenate([true_video[:, :frame_num], videos], axis=-3)  # (B, F, 512, W, 3)

    # ====================================================================
    #       Save videos in groups of 8  (EVERY 8 videos → 1 mp4)
    # ====================================================================
    B, F, H, W, _ = videos.shape
    group_size = 8
    num_groups = (B + group_size - 1) // group_size

    dataset_name = args.val_dataset_dir.split("/")[-1]
    output_dir = f"{args.output_path}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    model_name = pretrained_model_path.split("/")[-2]
    model_id = pretrained_model_path.split("/")[-1]

    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, B)
        group = videos[start:end]  # (gB, F, H, W, 3)

        # ---- Concatenate width-wise: (F, H, gB * W, 3)
        merged_video = np.concatenate([v for v in group], axis=2).astype(np.uint8)

        # ---- Generate filename ----
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{output_dir}/{time}_{model_name}_{model_id}_{args.start_idx}_part{g}.mp4"

        # ---- Write mp4 ----
        writer = imageio.get_writer(filename, fps=4)
        for frame in merged_video:
            writer.append_data(frame)
        writer.close()

    return

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, pipeline.vae, pipeline.unet

def main_eval(
    pretrained_model_path: str,
    clip_model_path: str,
    args: Dict,
    seed: Optional[int] = None,
):
    if seed is not None:
        set_seed(seed)
    # Load scheduler, tokenizer and models.
    pipeline, _, _ = load_primary_models(pretrained_model_path, eval=True)
    device = torch.device("cuda")
    pipeline.to(device)
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path,use_fast=False)
    text_encoder.requires_grad_(False).to(device)

    from video_dataset.video_transforms import Resize_Preprocess, ToTensorVideo
    preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess((256,256)), # 288 512
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    image_encoder = None
    if args.use_img_cond:
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
    id_list = args.val_idx

    for id in id_list:
        # prepare original instruction    
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
            if args.use_new_instru:
                text_id = args.new_instru
            # text_id = "Put the green block above the blue block."
            texts.append(text_id)

        # prepare ground-truth video
        video_path = anno['videos'][args.camera_idx]['video_path']
        video_path = f"{input_dir}/{video_path}"
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        try:
            true_video = vr.get_batch(range(length)).asnumpy()
        except:
            true_video = vr.get_batch(range(length)).numpy()
        true_video = true_video.astype(np.uint8)
        true_video = torch.from_numpy(true_video).permute(0, 3, 1, 2) # (l, c, h, w)
        if not args.only_one_clip:
            skip = args.skip_step
            start_idx = args.start_idx
            end_idx = start_idx + int(args.num_frames*skip)
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
        
        if args.use_img_cond:
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
    img_conds = torch.cat(img_conds, dim=0) if args.use_img_cond else None
    img_cond_masks = torch.tensor(img_cond_masks).to(device) if args.use_img_cond else None

    print("true_video size:",true_video.size())
    print("instructions:",texts)
    print("image condition mask:", img_cond_masks)
    eval(pipeline, texts, tokenizer, text_encoder, img_conds, img_cond_masks, image_encoder, true_videos, args, pretrained_model_path)
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
    val_args.num_inference_steps = args.num_inference_steps

    if args.val_idx is not None:
        idxs = args.val_idx.split("+")
        idxs = [int(idx) for idx in idxs]
        val_args.val_idx = idxs



    main_eval(pretrained_model_path=args.video_model_path,
              clip_model_path=args.clip_model_path,
        args=val_args,
    )

# bridge
# python make_prediction.py --eval --config video_conf/val_svd.yaml --video_model_path /cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-03-28T20-25-31/checkpoint-240000 --clip_model_path /cephfs/shared/llm/clip-vit-base-patch32 --val_dataset_dir video_dataset_instance/bridge --val_idx 2+10+8+14

# rt1
# python make_prediction.py --eval --config video_conf/val_svd.yaml --video_model_path /cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-03-28T20-25-31/checkpoint-240000 --clip_model_path /cephfs/shared/llm/clip-vit-base-patch32 --val_dataset_dir video_dataset_instance/rt1 --val_idx 1+6+8+10

# sthv2
# python make_prediction.py --eval --config video_conf/val_svd.yaml --video_model_path /cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-04-24T13-02-34/checkpoint-260000 --clip_model_path /cephfs/shared/llm/clip-vit-base-patch32 --val_dataset_dir video_dataset_instance/xhand  --val_idx 0+50+100+150

