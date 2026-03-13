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
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
# from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
# from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from diffusers import StableVideoDiffusionPipeline
from einops import rearrange, repeat
import imageio
import wandb
# from decord import VideoReader, cpu

from video_models.pipeline import MaskStableVideoDiffusionPipeline,TextStableVideoDiffusionPipeline

already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")

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

def finetune_unet(batch, accelerator, pipeline, unet, tokenizer, text_encoder, image_encoder,args,P_mean=0.7, P_std=1.6):
    pipeline.vae.eval()
    pipeline.image_encoder.eval()
    device = unet.device
    dtype = pipeline.vae.dtype
    vae = pipeline.vae
    # Convert videos to latent space
    pixel_values = batch['video']
    texts = batch['text']
    bsz, num_frames = pixel_values.shape[:2]

    # import pdb; pdb.set_trace()

    frames = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)
    if frames.shape[-3] == 3: # images
        latents = vae.encode(frames).latent_dist.mode() * vae.config.scaling_factor    
        latents = rearrange(latents, '(b f) c h w-> b f c h w', b=bsz)
        # enocde image latent
        image = pixel_values[:,0].to(dtype)
        noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
        image = image + noise_aug_strength * torch.randn_like(image)
        image_latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor
    else: # latents
        noise_aug_strength = 0.0
        latents = frames
        latents = rearrange(latents, '(b f) c h w-> b f c h w', b=bsz)
        image_latent = latents[:,0].to(dtype)


    condition_latent = repeat(image_latent, 'b c h w->b f c h w',f=num_frames)
    
    # clip text encoder output: encoder_hidden_states
    with torch.no_grad():
        img_cond = batch['img_cond'] if args.use_img_cond else None
        img_cond_mask = batch['img_cond_mask'] if args.use_img_cond else None
        encoder_hidden_states = encode_text(texts, tokenizer, text_encoder, img_cond, img_cond_mask, image_encoder, position_encode=args.position_encode, use_clip='clip' in args.clip_model_path, args=args)
        # for classifier-free guidance
        uncond_hidden_states = torch.zeros_like(encoder_hidden_states)
        text_mask = (torch.rand(encoder_hidden_states.shape[0], device=device)>0.05).unsqueeze(1).unsqueeze(2)

        encoder_hidden_states = encoder_hidden_states*text_mask+uncond_hidden_states*(~text_mask)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process) #[bsz, f, c, h , w]
    rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    c_skip = 1 / (sigma**2 + 1)
    c_out =  -sigma / (sigma**2 + 1) ** 0.5
    c_in = 1 / (sigma**2 + 1) ** 0.5
    c_noise = (sigma.log() / 4).reshape([bsz])
    loss_weight = (sigma ** 2 + 1) / sigma ** 2

    noisy_latents = latents + torch.randn_like(latents) * sigma
    input_latents = torch.cat([c_in * noisy_latents, condition_latent/vae.config.scaling_factor], dim=2)
    motion_bucket_id = args.motion_bucket_id
    fps = args.fps
    added_time_ids = pipeline._get_add_time_ids(fps, motion_bucket_id, 
        noise_aug_strength, encoder_hidden_states.dtype, bsz, 1, False)
    added_time_ids = added_time_ids.to(device)

    loss = 0

    accelerator.wait_for_everyone()
    model_pred = unet(input_latents, c_noise, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids).sample
    predict_x0 = c_out * model_pred + c_skip * noisy_latents 
    loss += ((predict_x0 - latents)**2 * loss_weight).mean()

    return loss


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
    vae.requires_grad_(False)
    
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

    from video_dataset.dataset_mix_v1 import Dataset_mix
    train_dataset = Dataset_mix(args,mode='train')
    val_dataset = Dataset_mix(args,mode='val')

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
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

    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, 
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
        run_name = train_args.run_name+output_dir.split('/')[-1]
        accelerator.init_trackers(train_args.project_name,config={}, init_kwargs={"wandb":{"name":run_name}})

    ########################################################################################
    # Start Train!

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    num_train_epochs = math.ceil(max_train_steps * gradient_accumulation_steps / len(train_dataloader))

    logger.info("***** Running training *****")
    logger.info(f"  accelerator.num_processes = {accelerator.num_processes}")
    logger.info(f"  gradient_accumulation_steps = {gradient_accumulation_steps}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    train_loss = 0.0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
     
    # *Potentially* Fixes gradient checkpointing training.
    # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    if kwargs.get('eval_train', False):
        unet.eval()
        text_encoder.eval()
 
    for epoch in range(first_epoch, num_train_epochs):
        # train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder):
                with accelerator.autocast():
                    # def finetune_unet(batch, accelerator, pipeline, unet, tokenizer, text_encoder,args,P_mean=0.7, P_std=1.6):
                    loss = finetune_unet(batch, accelerator, pipeline, unet, tokenizer,text_encoder,image_encoder, args)
                device = loss.device 
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                params_to_clip = unet.parameters()
                # accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                    
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if  global_step % 100 == 0 and global_step != 0:
                    accelerator.log({"train_loss": train_loss/100}, step=global_step)
                    logs = {"step_loss": train_loss/100, "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    train_loss = 0.0
                global_step += 1
                
                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    save_pipe(pretrained_model_path, global_step, accelerator, accelerator.unwrap_model(unet),accelerator.unwrap_model(text_encoder),vae, output_dir, is_checkpoint=True,save_pretrained_model=save_pretrained_model)

                if should_sample(global_step, validation_steps, validation_args):
                    unet.eval()
                    if global_step == 1: print("Performing validation prompt.")
                    # evaluate the model
                    val_loss_all = 0
                    with torch.no_grad():
                        for val_i, val_data in enumerate(val_dataloader):
                            if val_i >= validation_args.validation_num:
                                break
                            val_loss = validation(val_data, accelerator, pipeline, unet, tokenizer,text_encoder, image_encoder, args)
                            val_loss = accelerator.gather_for_metrics(val_loss)
                            val_loss_all += val_loss.mean().item()
                    val_loss_all /= val_i                  
                    accelerator.log({"validation_loss": val_loss_all}, step=global_step)

                    # sample some video for visualization
                    if accelerator.is_main_process:
                        print(f"validation loss: {val_loss_all}")
                        with accelerator.autocast():
                            for id in range(validation_args.video_num):
                                validate_video_generation(pipeline, tokenizer,text_encoder,image_encoder, val_dataset, validation_args, device, global_step,output_dir, id)
                    
                    unet.train()

            if global_step >= max_train_steps:
                break
  
    accelerator.end_training()



def validation(batch, accelerator, pipeline, unet, tokenizer, text_encoder, image_encoder, args,P_mean=0.7, P_std=1.6):
    pipeline.vae.eval()
    pipeline.image_encoder.eval()
    unet.eval()
    device = unet.device
    dtype = pipeline.vae.dtype
    vae = pipeline.vae
    # Convert videos to latent space
    pixel_values = batch['video']
    texts = batch['text']
    bsz, num_frames = pixel_values.shape[:2]

    # import pdb; pdb.set_trace()

    frames = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)
    if frames.shape[-3] == 3: # images
        latents = vae.encode(frames).latent_dist.mode() * vae.config.scaling_factor    
        latents = rearrange(latents, '(b f) c h w-> b f c h w', b=bsz)
        # enocde image latent
        image = pixel_values[:,0].to(dtype)
        noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
        image = image + noise_aug_strength * torch.randn_like(image)
        image_latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor
    else: # latents
        noise_aug_strength = 0.0
        latents = frames
        latents = rearrange(latents, '(b f) c h w-> b f c h w', b=bsz)
        image_latent = latents[:,0].to(dtype)


    condition_latent = repeat(image_latent, 'b c h w->b f c h w',f=num_frames)
    
    # clip text encoder output: encoder_hidden_states
    with torch.no_grad():
        img_cond = batch['img_cond'] if args.use_img_cond else None
        img_cond_mask = batch['img_cond_mask'] if args.use_img_cond else None
        encoder_hidden_states = encode_text(texts, tokenizer, text_encoder, img_cond, img_cond_mask, image_encoder, position_encode=args.position_encode, use_clip='clip' in args.clip_model_path, args=args)
        # for classifier-free guidance
        # uncond_hidden_states = torch.zeros_like(encoder_hidden_states)
        # text_mask = (torch.rand(encoder_hidden_states.shape[0], device=device)>0.05).unsqueeze(1).unsqueeze(2)

        # encoder_hidden_states = encoder_hidden_states*text_mask+uncond_hidden_states*(~text_mask)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process) #[bsz, f, c, h , w]
    rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    c_skip = 1 / (sigma**2 + 1)
    c_out =  -sigma / (sigma**2 + 1) ** 0.5
    c_in = 1 / (sigma**2 + 1) ** 0.5
    c_noise = (sigma.log() / 4).reshape([bsz])
    loss_weight = (sigma ** 2 + 1) / sigma ** 2

    noisy_latents = latents + torch.randn_like(latents) * sigma
    input_latents = torch.cat([c_in * noisy_latents, condition_latent/vae.config.scaling_factor], dim=2)

    motion_bucket_id = args.motion_bucket_id
    fps = args.fps
    added_time_ids = pipeline._get_add_time_ids(fps, motion_bucket_id, 
        noise_aug_strength, encoder_hidden_states.dtype, bsz, 1, False)
    added_time_ids = added_time_ids.to(device)

    loss = 0

    accelerator.wait_for_everyone()
    model_pred = unet(input_latents, c_noise, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids).sample
    predict_x0 = c_out * model_pred + c_skip * noisy_latents 
    loss += ((predict_x0 - latents)**2 * loss_weight).mean()

    return loss




def validate_video_generation(pipeline, tokenizer, text_encoder, image_encoder, val_dataset, args, device, train_steps, videos_dir, id,):
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

    # import pdb; pdb.set_trace()
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

    
    if true_video.shape[2] != 3:
        # decode latent
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])

    # import pdb; pdb.set_trace()
    true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
    true_video = true_video.detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

    new_videos = []
    for id_video, video in enumerate(videos):
        new_video = []
        for idx, frame in enumerate(video):
            new_video.append(np.array(frame))
            # print("frame",frame)
        new_videos.append(new_video)
    videos = new_videos

    videos = np.array([np.array(video) for video in videos]) #(2,16,256,256,3)
    print(videos.shape)
    videos = np.concatenate([true_video[:,:mask_frame_num],videos[:,mask_frame_num:]],axis=1)
    videos = np.concatenate([true_video,videos],axis=-3) #(2,16,512,256,3)
    videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)
    
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