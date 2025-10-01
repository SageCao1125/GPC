## bash eval.sh dual_bottles_pick_hard L515 100 300 0 4
## bash eval.sh pick_apple_messy L515 100 300 0 3
## bash eval.sh block_hammer_beat L515 100 300 0 7
## bash eval.sh empty_cup_place L515 100 300 0 2
## bash eval.sh dual_shoes_place L515 100 300 0 5
## bash eval.sh dual_bottles_pick_easy L515 100 300 0 0
## bash eval.sh shoe_place L515 100 300 0 5
## bash eval.sh blocks_stack_hard L515 100 300 0 4

DEBUG=False

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..
python ./script/eval_policy_dp.py "$task_name" "$head_camera_type" "$expert_data_num" "$checkpoint_num" "$seed"