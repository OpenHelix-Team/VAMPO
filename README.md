<div align="center">

# Dyn-VPP: Video Prediction Policy with Dynamic Optimization  

**Legend:** † Equal Contributions | ‡ Project Lead | * Corresponding Authors

---

**Zirui Ge<sup>1,12†</sup>** &nbsp;|&nbsp;
**Pengxiang Ding<sup>1,2,12†‡</sup>** &nbsp;|&nbsp;
**Yemin Wang<sup>3,12†</sup>** &nbsp;|&nbsp;
**Baohua Yin<sup>4,12</sup>** &nbsp;|&nbsp;
**Qishen Wang<sup>5,12</sup>**  

**Zhiyong Xie<sup>6</sup>** &nbsp;|&nbsp;
**Jinbo Wang<sup>13</sup>** &nbsp;|&nbsp;
**Hengtao Li<sup>7,12</sup>** &nbsp;|&nbsp;
**Runze Suo<sup>10,12</sup>** &nbsp;|&nbsp;
**Wenxuan Song<sup>9,12</sup>**  

**Han Zhao<sup>1,2,12</sup>** &nbsp;|&nbsp;
**Shangke Lyu<sup>9,12</sup>** &nbsp;|&nbsp;
**Haoang Li<sup>8</sup>** &nbsp;|&nbsp;
**Ran Cheng<sup>12‡</sup>** &nbsp;|&nbsp;
**Cheng Chi<sup>11</sup>**  

**Huibin Ge<sup>1</sup>** &nbsp;|&nbsp;
**Yaozhi Luo<sup>1*</sup>** &nbsp;|&nbsp;
**Donglin Wang<sup>2*</sup>**

---

### Affiliations

1. Zhejiang University  
2. Westlake University  
3. Xiamen University  
4. University of Sussex  
5. Tianjin University  
6. Wuhan University  
7. Hebei University of Technology  
8. HKUST (GZ)  
9. Nanjing University  
10. Fudan University  
11. Beijing Academy of Artificial Intelligence  
12. OpenHelix Robotics  
13. South China University of Technology
<a href='https://arxiv.org/abs/xxxx'><img src='https://img.shields.io/badge/ArXiv-XXXX-red'></a> 
<a href='https://dyn-vpp.github.io/Dyn-VPP'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

---

## 🚀 Overview

Video action models are a promising foundation for Vision–Language–Action (VLA) because they can learn rich visual dynamics directly from video. However, likelihood-oriented training of diffusion predictors emphasizes globally plausible futures and does not guarantee precision-critical visual dynamics needed for manipulation, so small prediction errors can be amplified by downstream policies.  

We propose **Dyn-VPP**, a post-training framework that casts multi-step denoising as policy optimization and aligns predicted future latents with expert visual dynamics via a verifiable terminal reward, without modifying any architecture. This enables explicit optimization of dynamics signals that are not captured by likelihood-only training. As a result, Dyn-VPP yields more accurate visual dynamics and improves downstream task execution. Experiments across diverse simulated and real-world manipulation settings show that Dyn-VPP achieves improved dynamics consistency and consistently higher task success.

<p>
    <img src="dyn_vpp_teaser.png" alt="method" width="100%" />
</p>

## 📌 Release Progress
- [x] Inference and evaluation code on Calvin
- [ ] Reinforcement learning post-training code

## 🛠️ Installation 
```bash
conda create -n dyn-vpp python==3.11
conda activate dyn-vpp

# Install calvin as described in (https://github.com/mees/calvin). 
git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh

# Install dyn-vpp requirements
cd ..
pip install -r requirements.txt
```


## 📷 CheckPoints 


| Ckpt name     | Training type | Size |
|---------------|------------------|---------|
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)  | CLIP text encoder, freezed during training        |  ~600M   |
| [Dyn-vpp_svd](https://huggingface.co/williammmgezju/Dyn-VPP_SVD)  | SVD video model trained by our method        | ~8G    |
| [Dyn-vpp_policy](https://huggingface.co/williammmgezju/Dyn-VPP_policy) |   Action model trained on annoted calvin abc dataset    |  ~1G  |


## 📊 Evaluation on Calvin abc benchmark
First, you need to follow instructions in the [officail calvin repo](https://github.com/mees/calvin) to install the calvin environments and download official calvin ABC-D dataset(about 500 G).

Next, download the [Dyn-vpp_svd](https://huggingface.co/williammmgezju/Dyn-VPP_SVD) video model and [Dyn-vpp_policy](https://huggingface.co/williammmgezju/Dyn-VPP_policy) action model. Set the video_model_folder and action_model_folder to the folder  where you save the model in the script.

```bash
bash scripts/eval_calvin.sh
```

## Acknowledgement

Dyn-VPP is developed from [Video prediction policy](https://github.com/roboterax/video-prediction-policy). We thank the authors for their efforts!



