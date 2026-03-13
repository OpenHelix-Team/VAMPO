
# import tensorflow_datasets as tfds
import cv2
import mediapy
import os
import random
import math
from diffusers.models import AutoencoderKL
# from decord import VideoReader, cpu
import mediapy
import torch
import numpy as np
import json
# vae = AutoencoderKL.from_pretrained("/home/disk2/gyj/llm/stable-diffusion-xl-base-1.0", subfolder="vae").to("cuda")
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
import mediapy
# vae = AutoencoderKL.from_pretrained("/cephfs/shared/llm/stable-diffusion-xl-base-1.0", subfolder="vae").to("cuda")

def load_hdf5(dataset_path):
    global compressed
    # dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)
        arm_qpos = root['/observations/arm_qpos'][()]
        hand_qpos = root['/observations/hand_qpos'][()]
        arm_end_pose = root['/observations/arm_end_pose'][()]
        waist_qpos = root['/observations/waist_qpos'][()]
        neck_qpos = root['/observations/neck_qpos'][()]
        action = root['/action'][()]
        text = root['/text'][()]
        # print(str(text))
        # print(str(text[0])[2:-1])
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
            # print(image_dict[cam_name][0].shape)
    cam_names = list(image_dict.keys())
    cam_names = sorted(cam_names)
    all_cam_videos = {}
    if compressed:
        for cam_name in cam_names:
            decompressed_image = np.array([cv2.imdecode(row,1) for row in image_dict[cam_name]])
            # print(decompressed_image.shape)
            all_cam_videos[cam_name] = decompressed_image
    else:
        for cam_name in cam_names:
            all_cam_videos[cam_name] = image_dict[cam_name]
        
    return arm_qpos, hand_qpos, arm_end_pose, waist_qpos, neck_qpos, action, all_cam_videos, text



############## raw data paths (teleoperation hdf5 files) ###############
raw_data_path = 'xhand'
# raw_data_path = ['']
# task_name = "xbot_0318_cloth"
# task_name = "xbot_0318"
task_name = "xbot_0417"
# task_name = "xbot_0407"
# xbot/opensource_robotdata/xbot_0318
dir = f'./xbot/opensource_robotdata/{task_name}'
############## saved paths ###############
video_dir = os.path.join(dir, 'videos')
latent_video_dir = os.path.join(dir, 'latent_videos')
anno_dir = os.path.join(dir, 'annotation')
os.makedirs(video_dir, exist_ok=True)
os.makedirs(latent_video_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)


import h5py
raw_file = []
subfolder = os.listdir(raw_data_path)
subfolder.sort()
# subfolder = [f for f in subfolder if 'cloths' in f]
# subfolder = [f for f in subfolder if 'cloths'not in f][:6]
subfolder = [f for f in subfolder if 'cloths'not in f][6:]
subfolder = ['1041601','1041701']
# subfolder = ['1042102']
# subfolder = ['1041001']
# subfolder = ['1040901', 'xbot_0408_v1']
# subfolder = ['1040301']
print(subfolder)
for sub in subfolder:
    sub_path = os.path.join(raw_data_path, sub)
    sub_raw_file = os.listdir(sub_path)
    sub_raw_file = [f for f in sub_raw_file if f.endswith('.hdf5')]
    sub_raw_file.sort()
    sub_raw_file = [os.path.join(sub_path, f) for f in sub_raw_file]
    raw_file += sub_raw_file

# raw_file = os.listdir(raw_data_path)
# raw_file = [f for f in raw_file if f.endswith('.hdf5')]
# raw_file.sort()
# raw_file = [os.path.join(raw_data_path, f) for f in raw_file]

####################################################
# start prepare vae latent data
vae = AutoencoderKLTemporalDecoder.from_pretrained("/home/disk2/gyj/llm/stable-video-diffusion-img2vid", subfolder="vae").to("cuda")

failed_num =0
success_num = 0
for file_num, file_name in enumerate(raw_file):
    # anno_ind_all = int(file_name.split('.')[-2].split('_')[-1])
    anno_ind_all = file_num
    data_type = 'val' if anno_ind_all%50==4 else 'train'
    with h5py.File(file_name, 'r') as file:
        # img = file['observations']['images']['cam_high'][:]
        arm_qpos, hand_qpos, arm_end_pose, waist_qpos, neck_qpos, action_all, image_dict,texts  = load_hdf5(file_name)
        print(file_name)
        # print(arm_qpos.shape, hand_qpos.shape, arm_end_pose.shape, action.shape, len(image_dict), image_dict.keys(), texts[1])
    text = str(texts[0])[2:-1]

    if 'capybara' in text:
        text = text.replace('capybara', 'pink capybara')
    if 'cube' in text:
        text = text.replace('cube', 'orange cube')
    if 'duck' in text:
        text = text.replace('duck', 'yellow duck')
    if 'mouse' in text:
        text = text.replace('mouse', 'brown mouse')
    if 'seal' in text:
        text = text.replace('seal', 'white seal')
    if 'bamboo' in text:
        p = random.random()
        if p < 0.5:
            text = text.replace('bamboo', 'green bamboo')
    text = 'place white bag in left and black bag in right'


    # split 1 trajectory into 5 trajectories if data is record at 50 hz. since the video model always predict 16 frames with frame intervel=0.1s
    skip_step = 5

    # for xarm+dexterous hand
    # states_all = np.concatenate((arm_end_pose,hand_qpos),axis=1)
    # action = [右臂，右手，左臂，左手] 观测都是先右后左的
    assert arm_end_pose.shape[-1] == 14
    assert hand_qpos.shape[-1] == 24
    states_all = np.concatenate((arm_end_pose[:,:7],hand_qpos[:,:12],arm_end_pose[:,7:],hand_qpos[:,12:]),axis=1)

    num_traj = 1 if data_type == 'val' else 1
    for j in range(num_traj):
        key_in_order = ['cam_high', 'cam_left', 'cam_right']
        latent_key = ['cam_high', 'cam_left', 'cam_right'] #['cam_high']
        action = action_all[j:]
        action = action[::skip_step]

        states = states_all[j:]
        states = states[::skip_step]
        # idx = k*5+j
        anno_ind = skip_step*anno_ind_all+j
        for idx,cam_name in enumerate(key_in_order):
            img_all = image_dict[cam_name]

            frame, h, w, c = img_all.shape
            pad_h = int(w*0.75)
            img_pad = np.zeros((frame, pad_h, w, c), dtype=np.uint8)
            img_pad[:, int(pad_h/2-h/2):int(pad_h/2+h/2), :w, :] = img_all

            img_all = img_pad

            img = img_all[j:]
            img = img[::skip_step]

            # crop
            frames = np.array(img)

            
            
            # save latent video
            latent_video_path = f"{dir}/latent_videos/{data_type}/{anno_ind}"
            os.makedirs(latent_video_path, exist_ok=True)
            frames = torch.tensor(frames).permute(0, 3, 1, 2).float().to("cuda") / 255.0*2-1
            # resize to 256*256
            x = torch.nn.functional.interpolate(frames, size=(256, 256), mode='bilinear', align_corners=False)
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            # save images to video
            video_path = f"{dir}/videos/{data_type}/{anno_ind}"
            os.makedirs(video_path, exist_ok=True)
            mediapy.write_video(f"{dir}/videos/{data_type}/{anno_ind}/{idx}.mp4", resize_video, fps=10)

            img_path = f"{dir}/imgs/{data_type}/{anno_ind}/{idx}.mp4"

            if cam_name in latent_key:
                with torch.no_grad():
                    batch_size = 64
                    latents = []
                    for i in range(0, len(x), batch_size):
                        batch = x[i:i+batch_size]
                        latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                        # x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                        latents.append(latent)
                    x = torch.cat(latents, dim=0)
                
                torch.save(x, f"{latent_video_path}/{idx}.pt")
        
        success_num += 1

        # print("text", "success!!!",anno_ind_all,"failed_num", failed_num, "success_num", success_num, action.shape)
        print("text", text, "num", file_num, "total_num", len(raw_file))
        # save anno
        info = {
            "task": "robot_trajectory_prediction",
            "texts": [
                text
            ],
            "videos": [
                {
                    "video_path": f"videos/{data_type}/{anno_ind}/0.mp4"
                },
                {
                    "video_path": f"videos/{data_type}/{anno_ind}/1.mp4"
                },
                {
                    "video_path": f"videos/{data_type}/{anno_ind}/2.mp4"
                }
            ],
            "episode_id": anno_ind,
            "video_length": len(action),
            "latent_videos": [
                {
                    "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/0.pt"
                },
                {
                    "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/1.pt"
                },
                {
                    "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/2.pt"
                },
                
            ],
            "states": states.tolist(),
            "actions": action.tolist(),
            }
        # write json file to /localssd/gyj/robo_datasets/opensource_robotdata/sthv2/annotation/train
        os.makedirs(f"{dir}/annotation/{data_type}", exist_ok=True)
        with open(f"{dir}/annotation/{data_type}/{anno_ind}.json", "w") as f:
            json.dump(info, f, indent=2)  


# running command
# CUDA_VISIBLE_DEVICES=2 python step1_prepare_latent_data.py