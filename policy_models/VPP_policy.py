import logging
from typing import Dict, Optional, Tuple
from functools import partial
from torch import einsum, nn
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import einops
from policy_models.edm_diffusion.score_wrappers import GCDenoiser

from policy_models.module.clip_lang_encoder import LangClip
from policy_models.edm_diffusion.gc_sampling import *
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_2D,Video_Former_3D
from diffusers import StableVideoDiffusionPipeline
from policy_models.module.diffusion_extract import Diffusion_feature_extractor
from transformers import AutoTokenizer, CLIPTextModelWithProjection


logger = logging.getLogger(__name__)

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.video_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


class VPP_Policy(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            use_text_not_embedding: bool = False,
            seed: int = 42,
            pretrained_model_path: str = '/cephfs/shared/gyj/ckpt/svd_pre/checkpoint-100000',
            text_encoder_path: str = '/home/disk2/gyj/hyc_ckpt/llm/clip-vit-base-patch32',
            use_position_encoding: bool = True,
            Former_depth: int = 3,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 1,
            num_latents: int = 3,
            use_Former: str = '3d',
            timestep: int = 20,
            max_length: int = 20,
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 7,
            action_seq_len: int = 10,
    ):
        super(VPP_Policy, self).__init__()
        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer
        self.use_position_encoding = use_position_encoding

        self.act_window_size = act_window_size
        self.action_dim = action_dim

        self.timestep = timestep
        self.extract_layer_idx = extract_layer_idx
        self.use_Former = use_Former
        self.Former_num_time_embeds = Former_num_time_embeds
        self.max_length = max_length

        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim

        if use_Former=='3d':
            self.Video_Former = Video_Former_3D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
                use_temporal=True,
             )
        elif use_Former == '2d':
            self.Video_Former = Video_Former_2D(
                    dim=latent_dim,
                    depth=Former_depth,
                    dim_head=Former_dim_head,
                    heads=Former_heads,
                    num_time_embeds=Former_num_time_embeds,
                    num_latents=num_latents,
                    condition_dim=condition_dim,
                 )
        else:
            self.Video_Former = nn.Linear(condition_dim,latent_dim)

        print('use_Former:', self.use_Former)
        print('use_all_layer',self.use_all_layer)

        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        # goal encoders
        self.language_goal = LangClip(model_name='ViT-B/32').to(self.device)

        pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(
            pretrained_model_path , eval = True)

        #text_encoder = CLIPTextModelWithProjection.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32")
        #tokenizer = AutoTokenizer.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32", use_fast=False)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)

        text_encoder = text_encoder.to(self.device).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.parameters():
            param.requires_grad = False

        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False

        pipeline = pipeline.to(self.device)
        pipeline.unet.eval()

        self.TVP_encoder = Diffusion_feature_extractor(pipeline=pipeline,
                                                        tokenizer=tokenizer,
                                                        text_encoder=text_encoder,
                                                        position_encoding = self.use_position_encoding)
        self.TVP_encoder = self.TVP_encoder.to(self.device)
        # policy network
        self.model = GCDenoiser(action_dim = action_dim,
                                obs_dim=latent_dim,
                                goal_dim=512,
                                num_tokens=num_latents,
                                goal_window_size = 1,
                                obs_seq_len = obs_seq_len,
                                act_seq_len = action_seq_len,
                                device=self.device,
                                sigma_data=0.5).to(self.device)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.ema_callback_idx = None

        for param in self.model.inner_model.proprio_emb.parameters():
            param.requires_grad = False
        for param in self.model.inner_model.goal_emb.parameters():
            param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def process_device(self):
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
        self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''
        optim_groups = [
            {"params": self.model.inner_model.parameters(),
             "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.Video_Former.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]


        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
            total_param_norm += p.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5

        self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)


    def training_step(self, dataset_batch: Dict[str, Dict],) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        total_loss, action_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)

        act_loss, sigmas, noise = self.diffusion_loss(
            predictive_feature,
            latent_goal,
            dataset_batch["actions"],
        )

        action_loss += act_loss
        total_loss += act_loss

        total_bs = dataset_batch["actions"].shape[0]

        self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(self, dataset_batch: Dict[str, Dict]) -> Dict[
        str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal module, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
            # Compute the required embeddings
        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)

        # predict the next action sequence
        action_pred = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            predictive_feature,
            latent_goal,
            inference=True,
        )
        dataset_batch["actions"] = dataset_batch["actions"].to(action_pred.device)
        # compute the mse action loss
        pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
        val_total_act_loss_pp += pred_loss

        output[f"idx:"] = dataset_batch["idx"]
        output["validation_loss"] = val_total_act_loss_pp
        return output

    def extract_predictive_feature(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'].to(self.device)
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'].to(self.device)
        # 3. we compute the language goal if the language modality is in the scope
        modality = "lang"
        if self.use_text_not_embedding:
            latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)

        language = dataset_batch["lang_text"]

        num_frames = self.Former_num_time_embeds
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
            language = language + language
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')
        perceptual_features = perceptual_features[:, :num_frames, :, :]
        #print('perceptual_features_shape:', perceptual_features.shape)

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=2)

        perceptual_features = perceptual_features.to(torch.float32)
        perceptual_features = self.Video_Former(perceptual_features)
        if self.use_Former=='linear':
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')
        predictive_feature = {'state_images': perceptual_features}
        predictive_feature['modality'] = modality
        return predictive_feature, latent_goal


    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)

    def diffusion_loss(
            self,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)

        x = torch.randn((len(latent_goal), self.act_window_size, self.action_dim), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self,batch):
        return self.training_step(batch)
        #def training_step(self, batch: Dict[str, Dict], batch_idx: int,
        #                  dataloader_idx: int = 0) -> torch.Tensor

    def eval_forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if 'lang_text' in goal:
            if self.use_text_not_embedding:
                # print(goal.keys())
                latent_goal = self.language_goal(goal["lang_text"])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                    obs["rgb_obs"]['rgb_static'].device)

        rgb_static = obs["rgb_obs"]['rgb_static']
        # rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        language = goal["lang_text"]

        num_frames = self.Former_num_time_embeds
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
            language = [language] + [language]
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')
        perceptual_features = perceptual_features[:, :num_frames, :, :]

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=2)

        perceptual_features = perceptual_features.to(torch.float32)
        perceptual_features = self.Video_Former(perceptual_features)
        if self.use_Former == 'linear':
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')

        perceptual_emb = {'state_images': perceptual_features}

        perceptual_emb['modality'] = "lang"
        #print('latent_goal_shape:',latent_goal.shape)
        #print('perceptual_features_shape:', perceptual_features.shape)

        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self.eval_forward(obs, goal)

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:

        self.model.to(dtype=self.dtype)

        self.Video_Former.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        #self.vae.to(dtype=self.dtype)
        self.TVP_encoder.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")


    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)