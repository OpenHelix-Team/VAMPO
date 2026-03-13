import torch
import torch.nn.functional as F
import lpips
from diffusers.models import AutoencoderKL
# from torchvision.models.optical_flow import raft_large
import torchvision
import os
import torchvision.transforms as tvf

# from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
# from mapanything.models import MapAnything

def pixel_reward_fn(
    pred_latents: torch.Tensor,
    true_latents: torch.Tensor,
    vae: AutoencoderKL,
    lpips_fn: lpips.LPIPS,
    lpips_weight: float = 1,
    mae_weight: float = 1,
    decode_chunk_size: int = 8,
    use_mae: bool = True,
    use_lpips: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = true_latents.device

    bsz = pred_latents.shape[0]
    num_frames = pred_latents.shape[1]

    all_rewards_mae = []
    all_rewards_lpips = []

    # Process in chunks to save memory
    with torch.no_grad():
        for i in range(0, bsz, decode_chunk_size):
            # Get chunks of latents
            pred_latents_chunk = pred_latents[i : i + decode_chunk_size]
            true_latents_chunk = true_latents[i : i + decode_chunk_size]
            
            chunk_bsz = pred_latents_chunk.shape[0]

            # Reshape for VAE decoder: (B, F, C, H, W) -> (B*F, C, H, W)
            pred_latents_reshaped = pred_latents_chunk.reshape(-1, *pred_latents_chunk.shape[2:]).to(vae.dtype)
            true_latents_reshaped = true_latents_chunk.reshape(-1, *true_latents_chunk.shape[2:]).to(vae.dtype)

            # Decode latents to images
            pred_images_decoded = vae.decode(pred_latents_reshaped / vae.config.scaling_factor, num_frames=num_frames).sample
            true_images_decoded = vae.decode(true_latents_reshaped / vae.config.scaling_factor, num_frames=num_frames).sample

            # # Reshape images to (B, F, C, H, W)
            # pred_images = pred_images_decoded.view(chunk_bsz, num_frames, *pred_images_decoded.shape[1:])
            # true_images = true_images_decoded.view(chunk_bsz, num_frames, *true_images_decoded.shape[1:])

            # # Save images for inspection
            # if not os.path.exists("inspection_images"):
            #     os.makedirs("inspection_images")
            # for frame_idx in range(num_frames):
            #     torchvision.utils.save_image(pred_images[0, frame_idx], f"inspection_images/pred_image_{i}_{frame_idx}.png")
            #     torchvision.utils.save_image(true_images[0, frame_idx], f"inspection_images/true_image_{i}_{frame_idx}.png")

            # --- Calculate MAE Reward ---
            if use_mae:
                mae = F.l1_loss(pred_images_decoded, true_images_decoded, reduction='none').mean(dim=[1, 2, 3])
                mae = mae.view(chunk_bsz, num_frames)
                reward_mae = - mae
                all_rewards_mae.append(reward_mae)

            # --- Calculate LPIPS Reward ---
            if use_lpips:
                lpips_dist = lpips_fn(pred_images_decoded, true_images_decoded).squeeze()
                if lpips_dist.ndim == 0:
                    lpips_dist = lpips_dist.unsqueeze(0)
                lpips_dist = lpips_dist.view(chunk_bsz, num_frames)
                reward_lpips = - lpips_dist
                all_rewards_lpips.append(reward_lpips)


        # Concatenate results from all chunks
    final_reward_mae = torch.cat(all_rewards_mae, dim=0) if all_rewards_mae else torch.zeros(bsz, num_frames, device=pred_latents.device)
    final_reward_lpips = torch.cat(all_rewards_lpips, dim=0) if all_rewards_lpips else torch.zeros(bsz, num_frames, device=pred_latents.device)

    # --- Combine Rewards ---
    final_reward = torch.zeros(bsz, num_frames, device=pred_latents.device)
    if use_lpips:
        final_reward += lpips_weight * final_reward_lpips

    if use_mae:
        final_reward += mae_weight * final_reward_mae

    return final_reward
    # return final_reward, final_reward_mae, final_reward_lpips, final_reward_depth


def latent_reward_fn(
    pred_latents: torch.Tensor,
    true_latents: torch.Tensor,
    vae: AutoencoderKL,
    decode_chunk_size: int = 8,
    use_mae: bool = True,
    use_cosine: bool = True,
    mae_weight: float = 1.0,
    cosine_weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    bsz = pred_latents.shape[0]
    num_frames = pred_latents.shape[1]

    all_rewards_mae = []
    all_rewards_cosine = []

    # Process in chunks to save memory
    with torch.no_grad():
        for i in range(0, bsz, decode_chunk_size):
            # Get chunks of latents (B, F, C, H, W)
            pred_latents_chunk = pred_latents[i : i + decode_chunk_size]
            true_latents_chunk = true_latents[i : i + decode_chunk_size]

            # --- Calculate MAE Reward ---
            if use_mae:
                mae = F.l1_loss(pred_latents_chunk, true_latents_chunk, reduction='none').mean(dim=[2, 3, 4])
                reward_mae = - mae  # Negative MAE as reward
                all_rewards_mae.append(reward_mae)

            # --- Calculate Cosine Similarity Reward ---
            if use_cosine:
                # Flatten spatial dimensions (C, H, W) for each token (B, F, C, H, W)
                pred_latents_flat = pred_latents_chunk.view(decode_chunk_size, num_frames, -1)  # (B, F, C*H*W)
                true_latents_flat = true_latents_chunk.view(decode_chunk_size, num_frames, -1)  # (B, F, C*H*W)

                # Normalize the latents for cosine similarity
                pred_latents_norm = F.normalize(pred_latents_flat, p=2, dim=-1)
                true_latents_norm = F.normalize(true_latents_flat, p=2, dim=-1)

                # Cosine similarity between predicted and true latents
                cosine_sim = torch.sum(pred_latents_norm * true_latents_norm, dim=-1)  # (B, F)

                # Reward based on cosine similarity
                reward_cosine = cosine_sim  # Higher similarity means higher reward
                all_rewards_cosine.append(reward_cosine)
            
    # Concatenate results from all chunks
    final_reward_mae = torch.cat(all_rewards_mae, dim=0) if all_rewards_mae else torch.zeros(bsz, num_frames, device=pred_latents.device)
    final_reward_cosine = torch.cat(all_rewards_cosine, dim=0) if all_rewards_cosine else torch.zeros(bsz, num_frames, device=pred_latents.device)
    # --- Combine Rewards ---
    final_reward = torch.zeros(bsz, num_frames, device=pred_latents.device)
    if use_mae:
        final_reward += mae_weight * final_reward_mae
    if use_cosine:
        # breakpoint()
        final_reward += cosine_weight * final_reward_cosine

    return final_reward
