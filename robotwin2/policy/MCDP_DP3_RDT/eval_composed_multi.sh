# Usage:
# bash policy/MCDP_DP3_RDT/eval_composed_multi.sh ${task_name} ${task_config} ${ckpt_setting} ${gpu_id} [config_path]
# Notes:
# - Three-policy eval: DP + DP3 + RDT
# - Enumerates valid (dp_w, dp3_w, rdt_w) with sum 1.0
# - dp3_w from 0.8 down to 0.1; remaining mass split between dp_w and rdt_w
# - Weights are multiples of 0.1
# Example:
#   bash policy/MCDP_DP3_RDT/eval_composed_multi.sh hanging_mug demo_clean demo_clean 0

#!/bin/bash
set -euo pipefail

policy_name=MCDP_DP3_RDT
task_name="$1"
task_config="$2"
ckpt_setting="${3:-$task_config}" 
gpu_id="${4:-0}"
config_path="${5:-policy/MCDP_DP3_RDT/deploy_policy.yml}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
cd ../..

log_dir="logs/composed_three/${task_name}/${task_config}"
mkdir -p "${log_dir}"
run_ts=$(date +%F-%H%M%S)

# Loop dp3_w from 0.8 down to 0.1
for dp3_w_int in {8..1}; do
  dp3_w=$(echo "scale=1; ${dp3_w_int} / 10" | bc -l)
  remaining_int=$((10 - dp3_w_int))
  
  # Split remaining mass between dp_w and rdt_w
  for dp_w_int in $(seq 1 $((remaining_int - 1))); do
    rdt_w_int=$((remaining_int - dp_w_int))
    
    dp_w=$(echo "scale=1; ${dp_w_int} / 10" | bc -l)
    rdt_w=$(echo "scale=1; ${rdt_w_int} / 10" | bc -l)
    
    echo "======================================================================"
    echo "task=${task_name}, config=${task_config}, ckpt_setting=${ckpt_setting}"
    echo "weights: dp_w=${dp_w}, dp3_w=${dp3_w}, rdt_w=${rdt_w}"
    echo "======================================================================"
    
    log_file="${log_dir}/dp_${dp_w}_dp3_${dp3_w}_rdt_${rdt_w}_${run_ts}.log"
    
    set +e
    PYTHONWARNINGS=ignore::UserWarning \
    python -u script/eval_three_compose_policies.py --config "${config_path}" \
      --overrides \
      --task_name "${task_name}" \
      --task_config "${task_config}" \
      --ckpt_setting "${ckpt_setting}" \
      --dp_w "${dp_w}" \
      --dp3_w "${dp3_w}" \
      --rdt_w "${rdt_w}" | tee "${log_file}"
    rc=$?
    set -e
    
    if [ $rc -ne 0 ]; then 
      echo "Run failed (dp_w=${dp_w}, dp3_w=${dp3_w}, rdt_w=${rdt_w}); see ${log_file}"
    fi
  done
done

echo "======================================================================"
echo "All weight-grid runs finished."
echo "Logs under: ${log_dir}"
echo "======================================================================"

