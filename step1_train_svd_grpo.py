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
from collections import defaultdict
import torch.distributed as dist

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
from functools import partial
tqdm = partial(tqdm, dynamic_ncols=True)
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from torch.utils.data.distributed import DistributedSampler

from diffusers.models import AutoencoderKL, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers import StableVideoDiffusionPipeline, EulerAncestralDiscreteScheduler
# from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
# from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from einops import rearrange, repeat
import imageio
import wandb
import lpips
# from decord import VideoReader, cpu

from video_models.pipeline import MaskStableVideoDiffusionPipeline, SamplingStableVideoDiffusionPipeline, ActorStableVideoDiffusionPipeline

already_printed_trainables = False
logger = get_logger(__name__, log_level="INFO")


def log_memory_usage(accelerator, logger, log_message):
    if accelerator.is_main_process:
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
        logger.info(f"--- MEMORY_LOG ({log_message}) ---")
        logger.info(f"  Allocated: {memory_allocated:.2f} MB / Reserved: {memory_reserved:.2f} MB")
        logger.info(f"  Max Allocated: {max_memory_allocated:.2f} MB / Max Reserved: {max_memory_reserved:.2f} MB")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16, variant='fp16')
    else:
        # pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
        
    return pipeline, pipeline.vae, pipeline.unet

def convert_svd(pretrained_model_path, out_path):
    pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet_mask", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    unet.conv_in.bias.data = copy.deepcopy(pipeline.unet.conv_in.bias)
    torch.nn.init.zeros_(unet.conv_in.weight)
    unet.conv_in.weight.data[:,1:]= copy.deepcopy(pipeline.unet.conv_in.weight)
    new_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path, unet=unet)
    new_pipeline.save_pretrained(out_path)

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    if unet_enable:
        unet.enable_gradient_checkpointing()
    else:
        unet.disable_gradient_checkpointing()
    if text_enable:
        text_encoder.gradient_checkpointing_enable()
    else:
        text_encoder.gradient_checkpointing_disable()

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }
    

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        for n, p in model.named_parameters():
            if p.requires_grad:
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)
    
    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, device, weight_dtype):
    for model in model_list:
        if model is not None: model.to(device, dtype=weight_dtype)

def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params =len(list(model.parameters()))
                    break
                    
                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True 
        print(f"{unfrozen_params} params have been unfrozen for training.")

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 5)  \
    and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        output_dir,
        is_checkpoint=False,
        save_pretrained_model=True
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    unet_out = copy.deepcopy(unet)
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        path, unet=unet_out).to(torch_dtype=torch.float32)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    logger.info(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline
    del unet_out
    torch.cuda.empty_cache()
    gc.collect()


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 


def prompt_image(image, processor, encoder):
    if type(image) == str:
        image = Image.open(image)
    image = processor(images=image, return_tensors="pt")['pixel_values']
    
    image = image.to(encoder.device).to(encoder.dtype)
    inputs = encoder(image).pooler_output.to(encoder.dtype).unsqueeze(1)
    #inputs = encoder(image).last_hidden_state.to(encoder.dtype)
    return inputs

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

def encode_text(texts, tokenizer, text_encoder, img_cond=None, img_cond_mask=None, img_encoder=None, position_encode=True, use_clip=False, args=None):
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

from grpo.pipeline_with_logprob_svd import pipeline_with_logprob_svd
from grpo.eas_with_logprob import eas_step_with_logprob
# from grpo.reward_fn import reward_fn
from grpo.reward_fn import latent_reward_fn

def compute_flowscale_weights(sigma_up, alpha=0.05, w_min=0.2, w_max=5.0, eps=1e-8):
    """
    sigma_up: (batch_size, timesteps, frames)
    returns w: (batch_size, timesteps, frames)
    """
    # Step 1: raw weight = (sigma^2 + eps)^(1/2)
    # (batch_size, timesteps, frames)
    w_raw = torch.sqrt(sigma_up**2 + eps)

    timesteps = w_raw.shape[1]  # 5 timesteps

    # Step 2: normalize so that the sum of weights along timesteps is 1 for each batch
    w_norm = w_raw * (timesteps / w_raw.sum(dim=1, keepdim=True))

    # Step 3: mix (avoid collapse)
    w_mix = (1 - alpha) * w_norm + alpha

    # Step 4: clip to keep the weights within [w_min, w_max]
    w_final = torch.clamp(w_mix, w_min, w_max)

    return w_final


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_args: Dict,
    shuffle: bool = True,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = None, # Eg: ("attn1", "attn2")
    extra_unet_params = None,
    extra_text_encoder_params = None,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    use_offset_noise: bool = False,
    rescale_schedule: bool = False,
    offset_noise_strength: float = 0.1,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    save_pretrained_model: bool = True,
    logger_type: str = 'tensorboard',
    num_generations: int = 4,
    num_inner_epochs: int = 1,
    clip_range: float = 0.2,
    adv_clip_max: float = 5.0,
    **kwargs
):
    #################################################################################
    # start accelerate
    *_, config = inspect.getargvalues(inspect.currentframe())
    args = train_args

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with='wandb',
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        print("set seed to", seed+accelerator.process_index)
        # set_seed(seed)
        set_seed(seed + accelerator.process_index)

    # Handle the output folder creation
    if accelerator.is_main_process:
        if args.use_lora:
           output_dir = output_dir+'peft_lora'
        output_dir = create_output_folders(output_dir, config)

    #################################################################################
    # load models

    # Load scheduler, tokenizer and models. The text encoder is actually image encoder for SVD
    pipeline, vae, unet = load_primary_models(pretrained_model_path)
    pipeline.to(accelerator.device)
    # pipeline.vae.to(accelerator.device)
    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # Initialize LPIPS model for reward calculation

    
    text_encoder, image_encoder = None, None
    if 'clip' in train_args.clip_model_path:
         # load clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        text_encoder = CLIPTextModelWithProjection.from_pretrained(train_args.clip_model_path)
        tokenizer = AutoTokenizer.from_pretrained(train_args.clip_model_path,use_fast=False)
        text_encoder.requires_grad_(False)

        if train_args.use_img_cond:
            # load image encoder
            from transformers import CLIPVisionModelWithProjection
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(train_args.clip_model_path)
            image_encoder.requires_grad_(False)
            image_encoder.to(unet.device)
    else:
        # Load t5 model directly
        print("load t5 model")
        from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(train_args.clip_model_path)
        # model = T5ForConditionalGeneration.from_pretrained("/cephfs/shared/llm/t5-v1_1-large").to("cuda")
        text_encoder = T5EncoderModel.from_pretrained(train_args.clip_model_path)

    text_encoder.to(accelerator.device)
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    trainable_modules_available = trainable_modules is not None

    # Unfreeze UNET Layers
    if trainable_modules_available:
        unet.train()
        handle_trainable_modules(
            unet, 
            trainable_modules, 
            is_enabled=True,
        )
    
    #################################################################################
    # if use lora, prepare lora model

    if args.use_lora:
        import peft
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from peft import prepare_model_for_kbit_training
        if args.lora_model_path:
            unet = PeftModel.from_pretrained(unet, args.lora_model_path, is_trainable=True)
        else:
            target_modules = ['to_k', 'to_q', 'to_v','out','proj','ff.net.','ff_in.net.','conv_out','conv_in']            
            modules_to_save = []
            peft_config = LoraConfig(
                r=args.lora_rank, lora_alpha=32, lora_dropout=0.05,
                bias="none",
                inference_mode=False,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )

            unet = get_peft_model(unet, peft_config)
        unet.print_trainable_parameters()

    #################################################################################
    # prepare optimizer

    optim_params = [
        param_optim(unet, trainable_modules_available, extra_params=extra_unet_params)
    ]

    params = create_optimizer_params(optim_params, learning_rate)
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    #################################################################################
    #Prepare Dataset

    from video_dataset.dataset_mix import Dataset_mix
    train_dataset = Dataset_mix(args,mode='train')
    val_dataset = Dataset_mix(args,mode='val')

    # from video_dataset.dataset_mix import Dataset_mix
    # train_dataset = Dataset_mix(args,mode='train')
    # val_dataset = Dataset_mix(args,mode='val')


    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=sample_batch_size,
        shuffle=shuffle
    )

    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=train_batch_size,
        shuffle=shuffle
    )

    validation_args = train_args # args
    ########################################################################################
    # Prepare everything with our `accelerator`.
    
    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet, 
        text_encoder, 
        True,
        True,
    )

    unet, text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, 
        text_encoder,
        optimizer, 
        train_dataloader, 
        val_dataloader,
        lr_scheduler, 
    )


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    if args.use_img_cond:
        models_to_cast.append(image_encoder)
    cast_to_gpu_and_type(models_to_cast, accelerator.device, weight_dtype)


    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_name = train_args.run_name + output_dir.split('/')[-1]
        accelerator.init_trackers(train_args.project_name, config={}, init_kwargs={"wandb": {"name": run_name}})

    ########################################################################################
    # Start GRPO Train!
    
    # GRPO-specific hyperparameters
    num_generations = args.num_generations
    num_inner_epochs = args.num_inner_epochs
    clip_range = args.clip_range
    adv_clip_max = args.adv_clip_max
    total_batch_size = sample_batch_size * accelerator.num_processes * gradient_accumulation_steps
    num_train_epochs = math.ceil(max_train_steps / len(train_dataloader))
    logger.info("***** Running GRPO training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {sample_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Number of generations per sample = {num_generations}")
    logger.info(f"  Number of inner training epochs = {num_inner_epochs}")

    global_step = 0
    first_epoch = 0
    for epoch in range(first_epoch, num_train_epochs):      

        for step, batch in enumerate(train_dataloader):        
            logger.info("***************** Start Sampling*******************")
            # log_memory_usage(accelerator, logger, f"Start of main loop step {step}") 
            pipeline.vae.eval()
            pipeline.unet.eval()
            pipeline.image_encoder.eval()

            true_latents = batch['latent'].to(accelerator.device)
            text = batch['text']

            # Get initial latents and embeddings
            image_latent = true_latents[:, 0]
            with torch.no_grad():
                img_cond, img_cond_mask = None, None
                text_token = encode_text(text, tokenizer, text_encoder, img_cond, img_cond_mask, image_encoder, position_encode=args.position_encode, use_clip='clip' in args.clip_model_path, args=args)
 
            sde_start = 0
            sde_end = 1
            train_num_timesteps = sde_end - sde_start
            # ---------- <<< 关键改动：确保 sampling 在 inference_mode + no_grad 下执行 >>> ----------
            with torch.inference_mode():                       # 比 torch.no_grad 更激进，会避免创建梯度图和部分缓冲
                with accelerator.autocast():                   # 你原来有 autocast，保留以节省显存/加速
                    pred_latents, latents_trajectory, log_probs_trajectory, sigma_ups_trajectory, added_time_ids_trajectory = \
                        SamplingStableVideoDiffusionPipeline.__call__(
                            pipeline,
                            image=image_latent,
                            text=text_token,
                            width=args.width,
                            height=args.height,
                            num_frames=args.num_frames,
                            num_inference_steps=args.num_inference_steps,
                            decode_chunk_size=args.decode_chunk_size,
                            num_videos_per_prompt=args.num_generations,
                            max_guidance_scale=args.guidance_scale,
                            fps=args.fps,
                            motion_bucket_id=args.motion_bucket_id,
                            sde_start=sde_start,
                            sde_end=sde_end,
                            output_type="latent",
                            mask=None
                        )


            latents_trajectory = [t.detach().cpu() for t in latents_trajectory]
            log_probs_trajectory = [t.detach().cpu() for t in log_probs_trajectory]
            sigma_ups_trajectory = [t.detach().cpu() for t in sigma_ups_trajectory]
            added_time_ids_trajectory = [t.detach().cpu() for t in added_time_ids_trajectory]

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
            with torch.inference_mode():
                all_rewards = latent_reward_fn(
                    pred_latents,
                    true_latents.repeat(num_generations, 1, 1, 1, 1), 
                    vae=vae
                )
            

            all_rewards = all_rewards.cpu()
            all_latents = torch.stack(latents_trajectory, dim=1)
            all_log_probs = torch.stack(log_probs_trajectory, dim=1)
            # all_sigma_ups = torch.stack(sigma_ups_trajectory, dim=1)
            all_added_time_ids = torch.stack(added_time_ids_trajectory, dim=1)

            try:
                del pred_latents, latents_trajectory, log_probs_trajectory, sigma_ups_trajectory, added_time_ids_trajectory
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()    

            timesteps = pipeline.scheduler.timesteps.repeat(sample_batch_size * num_generations, 1)
            # # Add image_latent to samples for training
            # samples_image_latent = image_latent.repeat(num_generations, 1, 1, 1)
            # all_weights = compute_flowscale_weights(all_sigma_ups) 


            samples = { 
                "image_latent": image_latent,
                "prompt_embeds": text_token,
                "timesteps": timesteps,
                "latents": all_latents[:, :-1],
                "next_latents": all_latents[:, 1:],
                "log_probs": all_log_probs,
                "added_time_ids": all_added_time_ids,
                "rewards": all_rewards,
                "image_latent": image_latent,
            }
            # # Calculate advantages
            # advantages = (samples["rewards"] - samples["rewards"].mean()) / (samples["rewards"].std() + 1e-8)
            # samples["advantages"] = advantages

            # rewards: [bsz * num_generations]
            rewards = samples["rewards"]
            rewards = rewards.reshape(sample_batch_size, num_generations, args.num_frames)
            advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)
            samples["advantages"] = advantages.reshape(sample_batch_size * num_generations, args.num_frames)

            # Offload samples to CPU
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    samples[k] = v.cpu()

            per_instance_rewards = samples['rewards'].mean(dim=-1) 


            grouped_rewards = per_instance_rewards.view(num_generations, sample_batch_size)


            group_mean = grouped_rewards.mean(dim=1)               

            logger.info(
                f"group-wise Metrics (Averaged over {num_generations} groups):\n"
                f"  Avg Group Max: {group_mean.max()}\n"
                f"  Avg Group Min: {group_mean.min()}\n"
                f"  Avg Group Mean: {group_mean.mean()}\n"
                f"  Avg Group Std: {group_mean.std()}"
            )

            
            #################### TRAINING ####################
            logger.info("***************** Start Training*******************")
            # log_memory_usage(accelerator, logger, "Start of Training section")
            pipeline.unet.train()
            info = defaultdict(list)
            # Rebatch for training
            total_samples, num_timesteps = samples["timesteps"].shape

            for k in ["prompt_embeds", "image_latent"]:
                if k in samples and samples[k].shape[0] < total_samples:
                    repeats = (num_generations,) + (1,) * (samples[k].dim() - 1)
                    samples[k] = samples[k].repeat(*repeats)
        
            samples_batched = {
                k: v.reshape(-1, train_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            total_loss_list = []

            for inner_epoch in range(num_inner_epochs):
                
                inner_epoch_loss = 0.0  
                
                with accelerator.accumulate(unet): 
                    
                    for i, sample in tqdm(
                        list(enumerate(samples_batched)),
                        desc=f"step {step}.{inner_epoch}: training",
                        position=0,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Move sample to device
                        for k, v in sample.items():
                            if isinstance(v, torch.Tensor):
                                sample[k] = v.to(accelerator.device)

                        # log_memory_usage(accelerator, logger, "Start of Training samples section")
                        # perms = torch.stack(
                        #     [
                        #         torch.randperm(num_timesteps, device=accelerator.device)
                        #         for _ in range(train_batch_size)
                        #     ]
                        # )
                        # for key in ["timesteps", "latents", "next_latents", "log_probs", "added_time_ids"]:
                        #     sample[key] = sample[key][
                        #         torch.arange(train_batch_size, device=accelerator.device)[:, None],
                        #         perms,
                        #     ]
                        
                        sample_loss = 0.0 
                        
                        for j in tqdm(
                            range(train_num_timesteps),
                            desc="Timestep",
                            position=1,
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                        ):
                            batch_latents = sample["latents"][:, j]
                            batch_next_latents = sample["next_latents"][:, j]
                            batch_log_probs = sample["log_probs"][:, j]
                            batch_timesteps = sample["timesteps"][:, j]
                            batch_added_time_ids = sample["added_time_ids"][:, j]
                            batch_prompt_embeds = sample["prompt_embeds"]
                            batch_advantages = sample["advantages"]
                            batch_image_latent = sample["image_latent"]

                            log_prob = ActorStableVideoDiffusionPipeline.__call__(
                                pipeline,
                                image=batch_image_latent,
                                text=batch_prompt_embeds,
                                batch_latents=batch_latents,
                                batch_next_latents=batch_next_latents,
                                batch_timesteps=batch_timesteps,
                                batch_added_time_ids=batch_added_time_ids,
                                width=args.width,
                                height=args.height,
                                num_frames=args.num_frames,
                                num_inference_steps=args.num_inference_steps,
                                decode_chunk_size=args.decode_chunk_size,
                                max_guidance_scale=args.guidance_scale,
                                fps=args.fps,
                                motion_bucket_id=args.motion_bucket_id,
                                mask=None
                            )
                            
                            advantages = torch.clamp(
                                batch_advantages,
                                -adv_clip_max,
                                adv_clip_max,
                            )

                            ratio = torch.exp(log_prob - batch_log_probs)
                            clipped_ratio = torch.clamp(
                                ratio,
                                1.0 - clip_range,
                                1.0 + clip_range,
                            )
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * clipped_ratio
                            
                            loss = torch.mean(torch.minimum(unclipped_loss, clipped_loss))
                            
                            # # 计算KL散度
                            # kl_divergence = torch.mean(log_prob - batch_log_probs)

                            # # 计算KL惩罚项
                            # kl_penalty = args.kl_penalty_factor * kl_divergence

                            # # 最终的损失函数
                            # loss = loss - kl_penalty

                            approx_kl = 0.5 * torch.mean((log_prob - batch_log_probs) ** 2)
                            clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
                            
                            accelerator.backward(loss) 
                            sample_loss += loss.item()

                            info["approx_kl"].append(approx_kl.item()) 
                            info["clipfrac"].append(clipfrac.item())
                            info["loss"].append(loss.item())

                        
                        current_sample_avg_loss = sample_loss / (train_num_timesteps)
                        
                        inner_epoch_loss += current_sample_avg_loss
                        
                        accelerator.log({
                            "sample_avg_loss": current_sample_avg_loss,
                        }, step=global_step)
                        logger.info(f"step={step}, inner_epoch={inner_epoch}, sample_i={i}")
                        logger.info(f"sample_avg_loss={current_sample_avg_loss:.6f}")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), config['max_grad_norm']
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                avg_epoch_loss = inner_epoch_loss / len(samples_batched) 
                total_loss_list.append(avg_epoch_loss)
                
                accelerator.log({
                    "inner_epoch_avg_loss": avg_epoch_loss,
                }, step=global_step)
                logger.info(f"total_loss_list={total_loss_list}")
                logger.info(f"inner_epoch={inner_epoch} finished. Avg Loss: {avg_epoch_loss:.6f}")
                
                
                # Clean up memory
                torch.cuda.empty_cache()
                gc.collect()

            if accelerator.sync_gradients:
                # log training-related stuff
                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_pipe(
                            pretrained_model_path,
                            global_step,
                            accelerator,
                            accelerator.unwrap_model(unet),
                            text_encoder,
                            vae,
                            output_dir,
                            is_checkpoint=True,
                            save_pretrained_model=save_pretrained_model
                        )
                
                if should_sample(global_step, validation_steps, validation_args):
                    unet.eval()
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            validate_video_generation(
                                pipeline,
                                tokenizer,
                                text_encoder,
                                image_encoder,
                                val_dataset,
                                args,
                                accelerator.device,
                                global_step,
                                output_dir,
                                accelerator.process_index,
                                pretrained_model_path,
                            )
                    unet.train()

            if global_step >= max_train_steps:
                break

            with torch.no_grad():
                # Aggressive cleanup
                # log_memory_usage(accelerator, logger, f"Before aggressive cleanup at end of step {step}")
                
                # Clear samples_batched, which is a list of dicts of tensors
                for sample_batch in samples_batched:
                    sample_batch.clear()
                del samples_batched
                
                # Clear samples, which is a dict of tensors
                samples.clear()
                del samples

                # Delete other large tensors created in the loop
                del all_rewards
                del all_latents, all_log_probs, all_added_time_ids
                del advantages, rewards, info, text_token, true_latents, image_latent, batch

                # Empty cache and collect garbage
                torch.cuda.empty_cache()
                gc.collect()
                # log_memory_usage(accelerator, logger, f"After aggressive cleanup at end of step {step}")


            
    accelerator.end_training()







def validate_video_generation(pipeline, tokenizer, text_encoder, image_encoder, val_dataset, args, device, train_steps, videos_dir, id, pretrained_model_path):
    videos_row = args.video_num if not args.debug else 1
    videos_col = 8
    batch_id = list(range(0,len(val_dataset),int(len(val_dataset)/videos_row/videos_col)))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    # random select 8 batch_id in len(val_dataset)
    # batch_id = np.random.choice(len(val_dataset),8)
    batch_list = [val_dataset.__getitem__(id, return_video = False) for id in batch_id]
    # actions = torch.cat([t['action'].unsqueeze(0) for i, t in enumerate(batch_list) ],dim=0).to(device, non_blocking=True)
    true_video = torch.cat([t['video'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    text = [t['text'] for i,t in enumerate(batch_list)]
    print("validation_text",text)
    
    mask_frame_num = 1
    image = true_video[:,0]
    with torch.no_grad():
        img_cond, img_cond_mask = None, None
        if args.use_img_cond:
            img_cond = torch.cat([t['img_cond'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device)
            print("img_cond",img_cond.shape)
            img_cond_mask = torch.tensor([t['img_cond_mask'] for i,t in enumerate(batch_list)]).to(device)
            print("img_cond",img_cond.shape, "img_cond_mask",img_cond_mask)
        text_token = encode_text(text, tokenizer, text_encoder, img_cond, img_cond_mask, image_encoder, position_encode=args.position_encode, use_clip='clip' in args.clip_model_path, args=args)
    
    with torch.no_grad():
        videos = MaskStableVideoDiffusionPipeline.__call__(
            pipeline,
            image=image,
            text=text_token,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None
        ).frames
        print("videos_num",len(videos))

    # --- Generate with original model for comparison ---
    original_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)

    with torch.no_grad():
        original_videos = MaskStableVideoDiffusionPipeline.__call__(
            original_pipeline,
            image=image,
            text=text_token,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None
        ).frames
    
    del original_pipeline
    torch.cuda.empty_cache()

    
    if true_video.shape[2] != 3:
        # decode latent
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            chunk = chunk.to(pipeline.vae.dtype)
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])

    true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
    true_video = true_video.detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

    new_videos = []
    for id_video, video in enumerate(videos):
        new_video = []
        for idx, frame in enumerate(video):
            new_video.append(np.array(frame))
        new_videos.append(new_video)
    videos = np.array(new_videos)

    new_original_videos = []
    for id_video, video in enumerate(original_videos):
        new_video = []
        for idx, frame in enumerate(video):
            new_video.append(np.array(frame))
        new_original_videos.append(new_video)
    original_videos = np.array(new_original_videos)

    # Replace first frame of generated videos with ground truth
    videos = np.concatenate([true_video[:, :mask_frame_num], videos[:, mask_frame_num:]], axis=1)
    original_videos = np.concatenate([true_video[:, :mask_frame_num], original_videos[:, mask_frame_num:]], axis=1)

    # Stack all three videos vertically (ground truth, fine-tuned, original)
    videos = np.concatenate([true_video, videos, original_videos], axis=-3)
    
    # Concatenate batches horizontally
    videos = np.concatenate([video for video in videos], axis=-2).astype(np.uint8)
    
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    writer = imageio.get_writer(filename, fps=4) # fps
    for frame in videos:
        writer.append_data(frame)
    writer.close()
    name = videos_dir.split('/')[-1]
    wandb.log({f"{name}_train_steps_{train_steps}": wandb.Video(filename, fps=4, format="mp4")})
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="video_conf/train_svd.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    cli_dict = OmegaConf.from_dotlist(args.rest)
    args_dict = OmegaConf.merge(args_dict, cli_dict)
    main(**args_dict)

# accelerate launch step1_train_svd.py --config video_conf/train_svd.yaml pretrained_model_path=/cephfs/cjyyj/code/video_robot_svd/output/svd/train_2025-05-05T01-35-37/checkpoint-20000
