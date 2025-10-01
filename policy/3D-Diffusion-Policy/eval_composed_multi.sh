# bash eval_composed_multi.sh dual_bottles_pick_hard L515 100 3000 0 4


first_run=true
# for loop
for dp_w in $(seq 0.1 0.1 0.9); do
    # compute dp3_w: dp_w + dp3_w = 1.0
    dp3_w=$(echo "1.0 - $dp_w" | bc)

    # print dp_w and dp3_w 
    echo "dp_w: $dp_w, dp3_w: $dp3_w"

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
    run_dir="./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/data/outputs/composed_${exp_name}_seed${seed}"

    DEBUG=False
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    if [ "$first_run" = true ]; then
        cd ../..
        first_run=false
    fi

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
done
