# Usage:
# bash policy/eval_composed_multi.sh ${task_name} ${task_config} ${gpu_id} [config_path]
# Notes:
# - Only task_name / task_config / gpu_id are required
# - DP / DP3 ckpt_setting / expert_data_num / checkpoint_num / seed come from YAML
# - If dp3_w is omitted in YAML, Python sets it to 1 - dp_w
policy_name=MCDP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
gpu_id=${4}
dp_w=${5}
dp3_w=${6}
config_path=${7:-policy/MCDP/deploy_policy.yml}

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "task=${task_name}, config=${task_config}, ckpt_setting=${ckpt_setting}, dp_w=${dp_w}, dp3_w=${dp3_w}"

cd ../..
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_mcdp_composed_policy.py --config ${config_path} \
                                      --overrides \
                                      --task_name ${task_name} \
                                      --task_config ${task_config} \
                                      --ckpt_setting ${ckpt_setting} \
                                      --dp_w ${dp_w} \
                                      --dp3_w ${dp3_w} 
