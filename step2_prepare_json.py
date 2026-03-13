import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from tqdm import tqdm
import torch
import random
import imageio
from decord import VideoReader, cpu
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
# from finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np
from scipy.spatial.transform import Rotation as R


def arm8_quat_to_euler(arm8):
    """
    arm8: (8,)
    [x, y, z, qx, qy, qz, qw, gripper]
    -> (7,)
    [x, y, z, roll, pitch, yaw, gripper]
    """
    pos = arm8[:3]
    quat = arm8[3:7]
    gripper = arm8[7:8]

    rpy = R.from_quat(quat).as_euler('xyz', degrees=False)

    return np.concatenate([pos, rpy, gripper])


def state16_quat_to_euler(state16):
    """
    (16,) -> (14,)
    """
    right = state16[:8]
    left = state16[8:]

    
    right7 = arm8_quat_to_euler(right)
    left7 = arm8_quat_to_euler(left)

    return np.concatenate([right7, left7])


def batch_state16_quat_to_euler(states16):
    """
    (N, 16) -> (N, 14)
    """
    return np.stack([state16_quat_to_euler(s) for s in states16])


def load_and_process_ann_file(data_root, ann_file, sequence_interval=1, start_interval=4, dataset_name='xhand_1024_v2', sequence_length=8):
    samples = []
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except:
        print(f'skip {ann_file}')
        return samples
    try:
        n_frames = len(ann['action'])
    except:
        n_frames = ann['video_length']

    # create multiple samples for robot data      
    # sequence_interval = 1
    # start_interval = 4
    # record idx for each clip
    base_idx = np.arange(0,sequence_length)*sequence_interval
    max_idx = np.ones_like(base_idx)*(n_frames-1)
    for start_frame in range(0,n_frames,start_interval):
        idx = base_idx + start_frame
        idx = np.minimum(idx,max_idx)
        idx = idx.tolist()
        if len(idx) == sequence_length:
            sample = dict()
            sample['dataset_name'] = dataset_name
            sample['ann_file'] = ann_file
            sample['episode_id'] = ann['episode_id']
            sample['frame_ids'] = idx
            sample['states'] = np.array(ann['states'])[idx[0]:idx[0]+1]
            sample['actions'] = np.array(ann['actions'])[idx]
            samples.append(sample)

    return samples

def init_anns(dataset_root, data_dir):
    final_path = f'{dataset_root}/{data_dir}'
    ann_files = [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith('.json')]
    # data_dir = f'{dataset_root}/{data_dir}'
    # ann_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    return ann_files

def init_sequences(data_root, ann_files, sequence_interval, start_interval, dataset_name,sequence_length):
    samples = []
    with ThreadPoolExecutor(32) as executor:
        future_to_ann_file = {executor.submit(load_and_process_ann_file, data_root, ann_file, sequence_interval, start_interval, dataset_name, sequence_length): ann_file for ann_file in ann_files}
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples
# start __main__

if __name__ == "__main__":
    dataset_names = 'fold_three_bowls+insert_phone_plug+pour_tea_into_cup+put_chain_in_the_box+put_vegetables_into_basket+stack_three_cubes+take_apart_lego'
    # dataset_names = 'fold_three_bowls'

    sequence_length = 16
    is_50hz = []
    trajs_each_demo = 1

    dataset_names = dataset_names.split('+')
    skip=1

    for data_type in ['val', 'train']:
        
        samples_all = []
        ann_files_all = []

        for dataset_name in dataset_names:
            data_save_path = "vpp_latent/opensource_robotdata"
            data_dir = f'annotation/{data_type}'
            data_root = f'{data_save_path}/{dataset_name}'
            if 'xhand_1125' in dataset_name:
                sequence_interval = int(skip*5)
                start_interval = 3
            else:
                sequence_interval = int(skip)
                start_interval = 1

            ann_files = init_anns(data_root, data_dir)
            if dataset_name in is_50hz:
                ann_files = [f for f in ann_files if int(f.split('/')[-1].split('.')[0])%trajs_each_demo == 0]
            ann_files_all.extend(ann_files)
            # print(ann_files)
            samples = init_sequences(data_root, ann_files,sequence_interval, start_interval, dataset_name, sequence_length)
            print(f'{dataset_name} {len(samples)} samples')
            samples_all.extend(samples)
        
        # # calculate the 1% and 99% perventile of the action and state for normalization
        # print("########################### state ###########################")
        # print(np.array(samples_all[0]['actions']).shape)
        # print(np.array(samples_all[0]['states']).shape)
        # state_all = [samples['states'] for samples in samples_all]
        # state_all = np.array(state_all)
        # print(state_all.shape)
        # state_all = state_all.reshape(-1, state_all.shape[-1])
        # # caculate the 1% and 99% of the action and state
        # state_01 = np.percentile(state_all, 1, axis=0)
        # state_99 = np.percentile(state_all, 99, axis=0)
        # print('state_01:', state_01)
        # print('state_99:', state_99)

        # print("########################### action ###########################")
        # action_all = [samples['actions']-samples['states'] for samples in samples_all]
        # action_all = np.array(action_all)
        # print(action_all.shape)
        # action_all = action_all.reshape(-1, action_all.shape[-1])
        # # caculate the 1% and 99% of the action and state
        # action_01 = np.percentile(action_all, 1, axis=0)
        # action_99 = np.percentile(action_all, 99, axis=0)
        # print('action_01:', action_01)
        # print('action_99:', action_99)

        print("########################### state ###########################")
        print(np.array(samples_all[0]['actions']).shape)
        print(np.array(samples_all[0]['states']).shape)

        # 收集所有 state
        state_all = [samples['states'] for samples in samples_all]  # 原版
        state_all = np.array(state_all)  # (num_traj, T, 16)
        print(state_all.shape)

        # flatten 所有 timestep
        state_all_flat = state_all.reshape(-1, state_all.shape[-1])  # (sum_T,16)

        # 🔧 转欧拉角
        state_all_euler = batch_state16_quat_to_euler(state_all_flat)  # (sum_T,14)

        # 计算 1% 和 99%
        state_01 = np.percentile(state_all_euler, 1, axis=0)
        state_99 = np.percentile(state_all_euler, 99, axis=0)
        print('state_01:', state_01)
        print('state_99:', state_99)

        # -------------------
        # 计算 action 的 percentile
        # -------------------
        print("########################### action ###########################")

        delta_actions_list = []
        for traj in samples_all:
            states = np.array(traj['states'])    # (T,16) 或 (1,16)
            actions = np.array(traj['actions'])  # (T,16)
            
            # 转欧拉角
            states_euler = batch_state16_quat_to_euler(states)
            actions_euler = batch_state16_quat_to_euler(actions)
            
            # delta action = action - state
            # 保持 timestep 对齐
            delta_action = actions_euler - states_euler
            delta_actions_list.append(delta_action)
            

        # concat 所有 trajectory
        action_all = np.concatenate(delta_actions_list, axis=0)  # (sum_T,14)
        print(action_all.shape)

        # 计算 1% 和 99%
        action_01 = np.percentile(action_all, 1, axis=0)
        action_99 = np.percentile(action_all, 99, axis=0)
        print('action_01:', action_01)
        print('action_99:', action_99)

        # remove state and action from samples
        for samples in samples_all:
            del samples['states']
            del samples['actions']

        import random
        random.shuffle(samples_all)
        print('step_num',data_type,len(samples_all))
        print('traj_num',data_type, len(ann_files_all))

        date = '0113_euler_all'
        # write to json file
        os.makedirs(f'{data_save_path}/annotation_all/{date}_interval{skip}/', exist_ok=True)
        with open(f'{data_save_path}/annotation_all/{date}_interval{skip}/{data_type}_all.json', 'w') as f:
            json.dump(samples_all, f, indent=4)
        
        stat = {
            'state_01': state_01.tolist(),
            'state_99': state_99.tolist(),
            'action_01': action_01.tolist(),
            'action_99': action_99.tolist()
        }
        with open(f'{data_save_path}/annotation_all/{date}_interval{skip}/{data_type}data.json', 'w') as f:
            json.dump(stat, f)