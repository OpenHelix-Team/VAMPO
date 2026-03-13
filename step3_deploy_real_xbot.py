from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
# import calvin_env

from typing import Union
import importlib
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
import numpy as np
# from pytorch_lightning import seed_everything
# from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import os
import imageio
from scipy.spatial.transform import Rotation as R  

class XbotAgent():
    def __init__(self):
        from hydra import compose, initialize
        from omegaconf import OmegaConf
        with initialize(config_path="conf", job_name="VPP_xbot_train.yaml"):
            cfg = compose(config_name="VPP_xbot_train.yaml")
        # print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
        self.cfg = cfg
        ckpt = cfg.ckpt_path
        state_dict = torch.load(ckpt, map_location='cpu')
        device = 'cuda:0'
        #print('state_dict_key:', state_dict.keys())
        cfg.model.device = device
        self.model = hydra.utils.instantiate(cfg.model)
        self.model.load_state_dict(state_dict['model'])
        # self.model
        self.model = self.model.to(device)
        self.model.eval()
        print('Model loaded')   
        args = cfg.dataset_args
        self.args = args
        self.a_min = np.array(args.action_01)[None,:]
        self.a_max = np.array(args.action_99)[None,:]
        self.s_min = np.array(args.state_01)[None,:]
        self.s_max = np.array(args.state_99)[None,:]
        print(f"action min: {self.a_min.shape}, action max: {self.a_max.shape}")

    def run_3image(self, image1, image2, image3, text, state):
        obs = dict()
        rgb_obs = {'rgb_static': image1, 'rgb_gripper': image2, 'rgb_gripper2': image3}
        obs['rgb_obs'] = rgb_obs

        state = self.normalize_bound(state, self.s_min, self.s_max)
        obs['state_obs'] =torch.from_numpy(state).float()

        goal = dict()
        goal["lang_text"] = text
        actions = self.model.step_real(obs, goal)

        actions = actions.detach().cpu().numpy()

        return actions

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def caculate_absolute_action(self, action_delta, state):
        action_delta = self.denormalize_bound(action_delta, self.a_min, self.a_max)

        return action_delta + state

    def process_action_xbot(self, label,frame_ids, rel = False):
        num_frames = len(frame_ids)
        frame_ids = frame_ids[:int(self.args.num_frames)] # (f)
        states = np.array(label['states'])[frame_ids] #(f, 38)
        command = np.array(label['actions'])[frame_ids]

        # print(f'states: {states.shape}, actions: {command.shape}')

        state = states[0:1] # current state

        a_dim = command.shape[-1]
        action_base = state[:,:a_dim] #(1,38)
        actions = command - action_base #(self.args.num_frames,38)

        # normalize
        action_scaled = self.normalize_bound(actions, self.a_min, self.a_max)
        state_scaled = self.normalize_bound(state, self.s_min, self.s_max)
        return torch.from_numpy(action_scaled).float(), torch.from_numpy(state_scaled).float()


from decord import VideoReader, cpu
def _load_video(video_path, frame_ids):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    assert (np.array(frame_ids) < len(vr)).all()
    assert (np.array(frame_ids) >= 0).all()
    vr.seek(0)
    frame_data = vr.get_batch(frame_ids).asnumpy()  # (frame, h, w, c)
    # central crop
    h, w = frame_data.shape[1], frame_data.shape[2]
    if h > w:
        margin = (h - w) // 2
        frame_data = frame_data[:, margin:margin + w]
    elif w > h:
        margin = (w - h) // 2
        frame_data = frame_data[:, :, margin:margin + h]
    # resize to 256x256
    # frame_data = torch.tensor(frame_data).permute(0, 3, 1, 2) # (l, c, h, w)
    # frame_data = torch.nn.functional.interpolate(frame_data, size=(256, 256), mode='bilinear', align_corners=False)

    return frame_data

if __name__ == "__main__":
    # @hydra.main(config_path="../conf", config_name="config_abc")

    agent = XbotAgent()
    # OmegaConf.to_yaml(cfg)
    test_dataset = '/localssd/gyj/opensource_robotdata/xbot_0407'
    test_idx = 1020 #770 #520
    video_file1 = f'{test_dataset}/videos/val/{test_idx}/0.mp4'
    video_file2 = f'{test_dataset}/videos/val/{test_idx}/1.mp4'
    video_file3 = f'{test_dataset}/videos/val/{test_idx}/2.mp4'
    json_file = f'{test_dataset}/annotation/val/{test_idx}.json'
    with open(json_file) as f:
        data = json.load(f)
        length = data["video_length"]
        states = data["states"]

    frames = _load_video(video_file1, list(range(length)))
    frames = frames.astype(np.uint8)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (l, c, h, w)
    frames = frames.float() / 255*2-1
    print('frames_all:', frames.shape)

    frames2 = _load_video(video_file2, list(range(length)))
    frames2 = frames2.astype(np.uint8)
    frames2 = torch.from_numpy(frames2).permute(0, 3, 1, 2) # (l, c, h, w)
    frames2 = frames2.float() / 255*2-1
    print('frames_all:', frames2.shape)

    frames3 = _load_video(video_file3, list(range(length)))
    frames3 = frames3.astype(np.uint8)
    frames3 = torch.from_numpy(frames3).permute(0, 3, 1, 2) # (l, c, h, w)
    frames3 = frames3.float() / 255*2-1
    print('frames_all:', frames3.shape)
    
    for idx in range(length-1):
        image1 = frames[idx:idx+1]
        image1 = image1.unsqueeze(0) # add batch size dim
        image2 = frames2[idx:idx+1]
        image2 = image2.unsqueeze(0)
        image3 = frames3[idx:idx+1]
        image3 = image3.unsqueeze(0)

        print(idx)
        with open(json_file) as f:
            data = json.load(f)
            text = data['texts'][0]
            true_action, _ = agent.process_action_xbot(data, [idx],rel=True)
            # print('text:', text)
            print('true_action:', true_action.shape)
            print('true_action:', true_action[0,0:7], true_action[0,19:26])
        with torch.no_grad():
            state = np.array(states)[idx:idx+1][None,:] # (1,1,38)
            print(image1.shape, image2.shape, image3.shape, text, state.shape)
            # torch.Size([1, 1, 3, 256, 256]) torch.Size([1, 1, 3, 256, 256]) torch.Size([1, 1, 3, 256, 256]) pick up the toy brown mouse and put it in the white box (1, 1, 38)
            
            #（1,1,3,256,256）(19)
            predicted_action = agent.run_3image(image1,image2,image3, text,state)
        print('predicted_action:', predicted_action.shape)
        print('predicted_action:', predicted_action[0,0,0:7], predicted_action[0,0,19:26])
        # print('true_action:', true_action)
        # print('true_action:', true_action)

        print(predicted_action[0].shape, state[0].shape)
        final_action = agent.caculate_absolute_action(action_delta=predicted_action[0], state=state[0])
        print('final_action:', final_action.shape)
        print('##################')