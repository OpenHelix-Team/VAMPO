from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import (
    EulerDiscreteScheduler,
    EulerDiscreteSchedulerOutput,
)

def index_for_multi_timestep(timestep, schedule_timesteps=None):
    # if schedule_timesteps is None:
    #     schedule_timesteps = self.timesteps
    timestep = timestep.to(schedule_timesteps.device).to(schedule_timesteps.dtype)
    mask = timestep.unsqueeze(1) == schedule_timesteps.unsqueeze(0)
    indices = mask.int().argmax(dim=1)

    return indices.reshape(timestep.shape)

def eas_step_with_logprob(
    self: EulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: torch.Tensor,
    sample: torch.FloatTensor,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Tuple:
    assert isinstance(self, EulerDiscreteScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    if prev_sample is None:
        sigma = self.sigmas[self.step_index]
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
    else:
        step_index = index_for_multi_timestep(timestep, self.timesteps)
        self.sigmas = self.sigmas.to(sample.device)
        sigma = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
        sigma_from = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
        sigma_to = self.sigmas[step_index + 1].view(-1, 1, 1, 1, 1)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)
    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    if self.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == "v_prediction":
        # * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    elif self.config.prediction_type == "sample":
        raise NotImplementedError("prediction_type not implemented yet: sample")
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )


    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma

    dt = sigma_down - sigma

    prev_sample_mean = sample + derivative * dt

    
    device = model_output.device
    
    if prev_sample is None:
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        prev_sample = prev_sample_mean + noise * sigma_up
        sigma_up_expanded = sigma_up.expand(model_output.shape[0], model_output.shape[1])
        self._step_index += 1
    else:
        sigma_up_expanded = sigma_up
    
    noise = (prev_sample - prev_sample_mean) / sigma_up
    # print("res: ", (prev_sample - prev_sample_mean).mean())
    # print("sigma_up: ", sigma_up.mean())
    # print("noise: ", noise.mean())
    log_prob_calculated = -0.5 * (noise ** 2) - torch.log(sigma_up) - 0.5 * math.log(2 * math.pi)


    # Aggregate based on the user's original placeholder logic (Mean over spatial dims)
    # Result shape: (Batch_Size, Channels) if input is (B, C, H, W)
    log_prob = log_prob_calculated.mean(dim=tuple(range(2, prev_sample.ndim)))

    # if prev_sample is None:
    #     sigma_up_expanded = sigma_up.expand(log_prob.shape[0], log_prob.shape[1])
    # else:
    #     sigma_up_expanded = sigma_up
    
    # upon completion increase step index by one
    return prev_sample.type(sample.dtype), log_prob, sigma_up_expanded

def stochastic_eds_step_with_logprob(
    self: EulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: torch.Tensor,
    sample: torch.FloatTensor,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    eta: float = 0.1,
) -> Tuple:
    assert isinstance(self, EulerDiscreteScheduler)

    if self.num_inference_steps is None:
        raise ValueError("Call set_timesteps first")

    if self.step_index is None:
        self._init_step_index(timestep)

    # ------------------------------------------------
    # σ bookkeeping (unchanged)
    # ------------------------------------------------
    if prev_sample is None:
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
    else:
        step_index = index_for_multi_timestep(timestep, self.timesteps)
        self.sigmas = self.sigmas.to(sample.device)
        sigma = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
        sigma_next = self.sigmas[step_index + 1].view(-1, 1, 1, 1, 1)

    sample = sample.to(torch.float32)

    # ------------------------------------------------
    # x0 prediction
    # ------------------------------------------------
    if self.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (
            model_output * (-sigma / torch.sqrt(sigma**2 + 1))
            + sample / (sigma**2 + 1)
        )
    else:
        raise ValueError("Unsupported prediction_type")

    # ------------------------------------------------
    # Drift (ODE derivative)
    # ------------------------------------------------
    derivative = (sample - pred_original_sample) / sigma

    delta_sigma = sigma_next - sigma  # negative
    drift = (1.0 - eta) * derivative * delta_sigma

    prev_sample_mean = sample + drift

    # ------------------------------------------------
    # Diffusion (Brownian term)
    # ------------------------------------------------
    diffusion_std = eta * torch.sqrt(torch.abs(delta_sigma))

    device = sample.device

    if prev_sample is None:
        noise = randn_tensor(
            sample.shape,
            dtype=sample.dtype,
            device=device,
            generator=generator,
        )
        prev_sample = prev_sample_mean + diffusion_std * noise
        self._step_index += 1
    else:
        noise = (prev_sample - prev_sample_mean) / diffusion_std

    # ------------------------------------------------
    # log p(x_{i-1} | x_i)
    # ------------------------------------------------
    log_prob_per_elem = (
        -0.5 * noise**2
        - torch.log(diffusion_std)
        - 0.5 * math.log(2 * math.pi)
    )

    log_prob = log_prob_per_elem.mean(dim=tuple(range(2, prev_sample.ndim)))

    return prev_sample.type(sample.dtype), log_prob, diffusion_std