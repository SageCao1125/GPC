<h1 align="center">
	[ICLR 2026] General Policy Composition (GPC)<br>
</h1>

<p align="center">
  <a href='https://arxiv.org/pdf/2510.01068'>
    <img src='https://img.shields.io/badge/Arxiv-Link-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
  </a>
  <a href='https://sagecao1125.github.io/GPC-Site/static/files/GPC_paper.pdf' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Paper-PDF-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Paper'>
  </a>
  <a href='https://sagecao1125.github.io/GPC-Site/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-green.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEiIHdpZHRoPSI2MDAiIGhlaWdodD0iNjAwIj48cGF0aCBkPSJNMTI5IDExMWMtNTUgNC05MyA2Ni05MyA3OEwwIDM5OGMtMiA3MCAzNiA5MiA2OSA5MWgxYzc5IDAgODctNTcgMTMwLTEyOGgyMDFjNDMgNzEgNTAgMTI4IDEyOSAxMjhoMWMzMyAxIDcxLTIxIDY5LTkxbC0zNi0yMDljMC0xMi00MC03OC05OC03OGgtMTBjLTYzIDAtOTIgMzUtOTIgNDJIMjM2YzAtNy0yOS00Mi05Mi00MmgtMTV6IiBmaWxsPSIjZmZmIi8+PC9zdmc+' alt='Project Page'>
  </a>
</p>


We propose General Policy Composition (GPC), a training-free method that enhances performance by combining the distributional scores of multiple pre-trained policies via a convex combination and test-time search. GPC is versatile, allowing for the plug-and-play composition of heterogeneous policies, including VA and VLA models, as well as those based on diffusion or flow-matching, irrespective of their input visual modalities.

## 📢 News
* **Jan 2026:** 🎉 GPC is accepted to **ICLR 2026**!

---
**Note**: This branch as well as the following guidelines are based on [RoboTwin-2.0](https://github.com/robotwin-Platform/RoboTwin), thanks for the authors' open resoure which greatly contributes to the community.

# 💻 Installation & 📚 Data Preparation

Please carefully follow the guidelines in [RoboTwin-2.0](https://robotwin-platform.github.io/doc/usage/index.html) for installation and environment setup.

To collect demonstrations in this branch, run the following command from the repository root. `task_config` is the RoboTwin task setting, such as `demo_clean`.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```


# 🧑🏻‍💻 Usage

This repository keeps the single policies and composed policies in separate top-level folders:

- `policy/DP`: image-based Diffusion Policy.
- `policy/DP3`: point-cloud-based 3D Diffusion Policy.
- `policy/RDT`: Robotics Diffusion Transformer.
- `policy/MCDP`: GPC composition of DP + DP3.
- `policy/MCDP_RDT`: GPC composition of DP + RDT.
- `policy/MCDP3_RDT`: GPC composition of DP3 + RDT.
- `policy/MCDP_DP3_RDT`: optional three-policy GPC composition of DP + DP3 + RDT.

The usual workflow is:

1. Collect demonstrations with RoboTwin.
2. Train or prepare the single-policy checkpoints.
3. Edit the composed policy YAML file to point to the checkpoints.
4. Run the composed-policy evaluation script from the corresponding composed-policy folder.

## Step 1. Collect Demonstrations

Run from the repository root:

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```

Here `task_config` is the RoboTwin task setting, for example `demo_clean`.

## Step 2. Prepare Single Policies

### 1. Diffusion Policy (DP)

```
cd policy/DP
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
```

For single-policy evaluation:

```
cd policy/DP
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
```

The composed policies load DP checkpoints from:

```
policy/DP/checkpoints/${task_name}-${dp_ckpt_setting}-${dp_expert_data_num}-${dp_seed}/${dp_checkpoint_num}.ckpt
```

### 2. 3D Diffusion Policy (DP3)

```
cd policy/DP3
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
```

If you want to train DP3 with RGB point-cloud color:

```
cd policy/DP3
bash train_rgb.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
```

For single-policy evaluation:

```
cd policy/DP3
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
```

The composed policies load DP3 checkpoints from one of the following paths:

```
policy/DP3/checkpoints/${task_name}-${dp3_ckpt_setting}-${dp3_expert_data_num}_${dp3_seed}/${dp3_checkpoint_num}.ckpt
policy/DP3/checkpoints/${task_name}-${dp3_ckpt_setting}-${dp3_expert_data_num}_w_rgb_${dp3_seed}/${dp3_checkpoint_num}.ckpt
```

For the DP + DP3 composition (`policy/MCDP`), use the second path when RGB point-cloud color is enabled. The current RDT-related compositions (`policy/MCDP3_RDT` and `policy/MCDP_DP3_RDT`) use the non-RGB DP3 path by default. If your DP3 training run writes checkpoints under `policy/DP3/3D-Diffusion-Policy/checkpoints/`, copy or symlink the selected checkpoint directory into `policy/DP3/checkpoints/` before running composed evaluation.

### 3. Robotics Diffusion Transformer (RDT)

```
cd policy/RDT
bash process_data_rdt.sh ${task_name} ${task_config} ${expert_data_num} ${gpu_id}
bash generate.sh ${config_name}
bash finetune.sh ${config_name}
```

For single-policy evaluation:

```
cd policy/RDT
bash eval.sh ${task_name} ${task_config} ${model_name} ${checkpoint_id} ${seed} ${gpu_id}
```

The composed RDT policies currently expect the RDT checkpoint at:

```
policy/RDT/checkpoints/rdt_${task_name}/checkpoint-${rdt_checkpoint_num}/pytorch_model/mp_rank_00_model_states.pt
```

Therefore, when using the provided composed evaluation scripts, make sure the RDT model directory is named `rdt_${task_name}`. The current composed evaluation scripts set this name internally; changing `rdt_ckpt_setting` in a YAML file will not change the loaded RDT directory unless you also modify the script.

**Important**: For distribution-level composition, the composed policies should share an aligned action distribution space. In practice, keep the action dimension, action horizon, scheduler family, prediction type, and inference steps consistent across the policies being composed.

## Step 3. Configure GPC Checkpoints

Before running GPC evaluation, edit the corresponding YAML file:

- DP + DP3: `policy/MCDP/deploy_policy.yml`.
- DP + RDT: `policy/MCDP_RDT/deploy_policy.yml`.
- DP3 + RDT: `policy/MCDP3_RDT/deploy_policy.yml`.
- DP + DP3 + RDT: `policy/MCDP_DP3_RDT/deploy_policy.yml`.

The YAML files specify checkpoint settings such as:

```
dp_ckpt_setting: demo_clean
dp_expert_data_num: 50
dp_checkpoint_num: 600
dp_seed: 0

dp3_ckpt_setting: demo_clean
dp3_expert_data_num: 50
dp3_checkpoint_num: 3000
dp3_seed: 0
dp3_use_rgb: false  # used by DP + DP3 composition; RDT-related compositions use non-RGB DP3 by default

rdt_checkpoint_num: 5000
rdt_step: 32
```

The `ckpt_setting` argument in the shell commands is kept for command compatibility. For composed policies, the actual DP/DP3 checkpoint choices are mainly controlled by the `dp_*` and `dp3_*` fields in the corresponding `deploy_policy.yml`. The provided RDT-composition scripts use `rdt_${task_name}` as the RDT model directory name, regardless of any `rdt_ckpt_setting` value in YAML.

## Step 4. Compose Policies with GPC

Run the composed evaluation scripts from inside each composed-policy folder. The scripts will `cd` back to the repository root internally.

### 1. DP + DP3

Configuration file: `policy/MCDP/deploy_policy.yml`.

Run one evaluation with explicit weights:

```
cd policy/MCDP
bash eval_composed.sh ${task_name} ${task_config} ${ckpt_setting} ${gpu_id} ${dp_w} ${dp3_w}
```

Run a weight sweep from `0.1` to `0.9`:

```
cd policy/MCDP
bash eval_composed_multi.sh ${task_name} ${task_config} ${ckpt_setting} ${gpu_id}
```

### 2. DP + RDT

Configuration file: `policy/MCDP_RDT/deploy_policy.yml`.

Run a weight sweep over `dp_w` and `rdt_w`:

```
cd policy/MCDP_RDT
bash eval_composed_multi.sh ${task_name} ${task_config} ${ckpt_setting} ${gpu_id}
```

### 3. DP3 + RDT

Configuration file: `policy/MCDP3_RDT/deploy_policy.yml`.

Run a weight sweep over `dp3_w` and `rdt_w`:

```
cd policy/MCDP3_RDT
bash eval_composed_multi.sh ${task_name} ${task_config} ${ckpt_setting} ${gpu_id}
```

### 4. DP + DP3 + RDT

Configuration file: `policy/MCDP_DP3_RDT/deploy_policy.yml`.

Run a grid sweep over valid `dp_w`, `dp3_w`, and `rdt_w` values whose sum is `1.0`:

```
cd policy/MCDP_DP3_RDT
bash eval_composed_multi.sh ${task_name} ${task_config} ${ckpt_setting} ${gpu_id}
```

The evaluation results are saved under:

```
eval_result/${task_name}/${composed_policy_name}/${task_config}/
```

Logs from the sweep scripts are saved under `logs/`.



# 😺 Acknowledgement
Our code is generally built upon: [RoboTwin](https://github.com/robotwin-Platform/RoboTwin), [3D Diffusion Poliy](https://github.com/YanjieZe/3D-Diffusion-Policy), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy),[RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer). We thank all these authors for their nicely open sourced code and their great contributions to the community. 

# 👍 Citation
```
@article{cao2025gpc,
      title={Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition}, 
      author={Jiahang Cao and Yize Huang and Hanzhong Guo and Rui Zhang and Mu Nan and Weijian Mai and Jiaxu Wang and Hao Cheng and Jingkai Sun and Gang Han and Wen Zhao and Qiang Zhang and Yijie Guo and Qihao Zheng and Chunfeng Song and Xiao Li and Ping Luo and Andrew F. Luo},
      journal={arXiv preprint arXiv:2510.01068},
      year={2025},
      url={https://arxiv.org/abs/2510.01068}, 
}
```

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

