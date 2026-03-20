<div align="center">

# VAMPO: Policy Optimization for Improving Visual Dynamics in Video Action Models

<a href='https://arxiv.org/abs/xxxx'><img src='https://img.shields.io/badge/ArXiv-XXXX-red'></a> 
<a href='https://vampo-robot.github.io/VAMPO/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 
<a href='https://huggingface.co/williammmgezju'><img src='https://img.shields.io/badge/🤗-HuggingFace-yellow'></a>
</div>

---

## 🚀 Overview

Video action models are an appealing foundation for Vision–Language–Action systems because they
can learn visual dynamics from large-scale video data and transfer this knowledge to downstream robot
control. Yet current diffusion-based video predictors are trained with likelihood-surrogate objectives,
which encourage globally plausible predictions without explicitly optimizing the precision-critical
visual dynamics needed for manipulation. This objective mismatch often leads to subtle errors in
object pose, spatial relations, and contact timing that can be amplified by downstream policies. We
propose VAMPO, a post-training framework that directly improves visual dynamics in video action
models through policy optimization. Our key idea is to formulate multi-step denoising as a sequential
decision process and optimize the denoising policy with rewards defined over expert visual dynamics in
latent space. To make this optimization practical, we introduce an Euler Hybrid sampler that injects
stochasticity only at the first denoising step, enabling tractable low-variance policy-gradient estimation
while preserving the coherence of the remaining denoising trajectory. We further combine this design
with GRPO and a verifiable non-adversarial reward based on L1 distance and cosine similarity. Across
diverse simulated and real-world manipulation tasks, VAMPO improves task-relevant visual dynamics,
leading to better downstream action generation and stronger generalization.

<p>
    <img src="teaser.png" alt="method" width="100%" />
</p>

## 📌 Release Progress
- [x] Inference and evaluation code on Calvin
- [ ] Reinforcement learning post-training code

## 🛠️ Installation 
```bash
conda create -n VAMPO python==3.11
conda activate VAMPO

# Install calvin as described in (https://github.com/mees/calvin). 
git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh

# Install VAMPO requirements
cd ..
pip install -r requirements.txt
```


## 📷 CheckPoints 


| Ckpt name     | Training type | Size |
|---------------|------------------|---------|
| [VAMPO_svd](https://huggingface.co/williammmgezju/VAMPO_SVD)  | SVD video model trained by our method        | ~8G    |
| [VAMPO_policy](https://huggingface.co/williammmgezju/VAMPO_policy) |   Action model trained on annoted calvin abc dataset    |  ~1G  |
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)  | CLIP text encoder       |  ~600M   |


## 📊 Evaluation on Calvin abc benchmark
First, you need to follow instructions in the [officail calvin repo](https://github.com/mees/calvin) to install the calvin environments and download official calvin ABC-D dataset(about 500 G).

Next, download the [VAMPO_svd](https://huggingface.co/williammmgezju/VAMPO_SVD) video model and [VAMPO_policy](https://huggingface.co/williammmgezju/VAMPO_policy) action model. Set the video_model_folder and action_model_folder to the folder  where you save the model in the script.

```bash
bash scripts/eval_calvin.sh
```

## Acknowledgement

Dyn-VPP is developed from [Video prediction policy](https://github.com/roboterax/video-prediction-policy). We thank the authors for their efforts!



