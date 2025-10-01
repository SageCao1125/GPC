# bash eval.sh mug_hanging 10 3000 0 0
# bash eval.sh dual_bottles_pick_hard L515 100 3000 0 5
# bash eval.sh pick_apple_messy L515 100 3000 0 1
# bash eval.sh block_hammer_beat L515 100 3000 0 6
# bash eval.sh empty_cup_place L515 100 3000 0 1
# bash eval.sh dual_shoes_place L515 100 3000 0 4
# bash eval.sh dual_bottles_pick_easy L515 100 3000 0 1
# bash eval.sh shoe_place L515 100 3000 0 4
# bash eval.sh blocks_stack_hard L515 100 3000 0 2

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}
alg_name=robot_dp3_ddpm
config_name=${alg_name}
addition_info=eval
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/data/outputs/${exp_name}_seed${seed}"

DEBUG=False
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..
python script/eval_policy_dp3.py --config-name=${config_name}.yaml \
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
                            head_camera_type=${head_camera_type}