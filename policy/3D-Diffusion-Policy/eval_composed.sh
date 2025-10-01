# bash eval_composed.sh dual_bottles_pick_hard L515 100 3000 0 6 0.6 0.4

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}
dp_w=${7}
dp3_w=${8}
alg_name=robot_dp3_ddpm
config_name=${alg_name}
addition_info=eval
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/data/outputs/composed_${exp_name}_seed${seed}"

DEBUG=False
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "dp_w: $dp_w, dp3_w: $dp3_w"

cd ../..
python script/eval_policy_composed_policy.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            raw_task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint_num=${checkpoint_num} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=${head_camera_type} \
                            dp_w=${dp_w} \
                            dp3_w=${dp3_w}