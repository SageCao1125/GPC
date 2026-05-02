# Usage:
# bash policy/eval_composed_multi.sh ${task_name} ${task_config} ${gpu_id} [config_path]

# task_name=${1}
# task_config=${2}
# gpu_id=${3}
# config_path=${4:-policy/MCDP/deploy_policy.yml}

# export CUDA_VISIBLE_DEVICES=${gpu_id}
# cd ../..

# for dp_w in $(seq 0.1 0.1 0.9); do
#     dp3_w=$(echo "1.0 - $dp_w" | bc)
#     echo "task=${task_name}, config=${task_config}, dp_w=${dp_w}, dp3_w=${dp3_w}"
#     PYTHONWARNINGS=ignore::UserWarning \
#     python script/eval_composed_policy.py --config ${config_path} \
#       --overrides \
#       --task_name ${task_name} \
#       --task_config ${task_config} \
#       --dp_w ${dp_w} \
#       --dp3_w ${dp3_w} 
# done


# Usage:
# bash policy/eval_composed_multi.sh ${task_name} ${task_config} ${gpu_id} [config_path]
# Notes:
# - Only task_name / task_config / gpu_id are required (batch eval for multiple tasks/configs)
# - DP / DP3 ckpt_setting / expert_data_num / checkpoint_num / seed come from YAML
# - If dp3_w is omitted in YAML, Python sets it to 1 - dp_w
#bash eval_composed_multi.sh hanging_mug demo_clean demo_clean 0

#!/bin/bash
set -euo pipefail
policy_name=MCDP_RDT
task_name="$1"
task_config="$2"
ckpt_setting="${3:-$task_config}" 
gpu_id="${4:-0}"
config_path="${5:-policy/MCDP_RDT/deploy_policy.yml}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
cd ../..

log_dir="logs/composed/${task_name}/${task_config}"
mkdir -p "${log_dir}"
run_ts=$(date +%F-%H%M%S)

for rdt_w in $(seq 0.1 0.1 0.9); do
  dp_w=$(echo "1.0 - ${rdt_w}" | bc)
  echo "task=${task_name}, config=${task_config}, ckpt_setting=${ckpt_setting}, rdt_w=${rdt_w}, dp_w=${dp_w}"
  log_file="${log_dir}/rdt_${rdt_w}_dp_${dp_w}_${run_ts}.log"

  set +e
  PYTHONWARNINGS=ignore::UserWarning \
  python -u script/eval_dp_composed_policy.py --config "${config_path}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --rdt_w "${rdt_w}" \
    --dp_w "${dp_w}" | tee "${log_file}"
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then echo "Run failed (rdt_w=${rdt_w}); see ${log_file}"; fi
done

