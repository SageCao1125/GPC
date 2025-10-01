# bash eval.sh mug_hanging 10 3000 0 0


task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}
alg_name=robot_dp3
config_name=${alg_name}
addition_info=eval
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/data/outputs/${exp_name}_seed${seed}"

DEBUG=False
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
                            head_camera_type=${head_camera_type} \
                            policy.use_pc_color=True
