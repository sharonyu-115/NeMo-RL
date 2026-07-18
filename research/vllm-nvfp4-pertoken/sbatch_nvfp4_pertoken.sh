#!/bin/bash
#SBATCH --job-name=general_sa-nemo_rl.nvfp4-pertoken-probe
#SBATCH --account=general_sa
#SBATCH --partition=batch,tcpo,36x2-a01r,a02grace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/nvfp4-pertoken-%j.out
#SBATCH --error=logs/nvfp4-pertoken-%j.err

# =============================================================================
# Standalone validation of vLLM per-token NVFP4 activation scaling (vllm#48538)
# before wiring into NeMo-RL (on top of NVIDIA-NeMo/RL#2983). See README.md.
#
# Runs pytest checks A-D from test_nvfp4_pertoken.py inside the vLLM nightly
# container. Select checks with RUN_CHECKS (comma list of pytest -k exprs),
# default runs everything in dependency order.
#   RUN_CHECKS="check_a or check_b" sbatch sbatch_nvfp4_pertoken.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
fi
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE in .env}"
: "${MODEL_BF16:?set MODEL_BF16 in .env}"
: "${MODEL_NVFP4:?set MODEL_NVFP4 in .env}"
: "${MODEL_SMALL:?set MODEL_SMALL in .env}"

RUN_CHECKS="${RUN_CHECKS:-check_a or check_b or check_c or check_d}"
RESULTS_DIR="${SCRIPT_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" "${SCRIPT_DIR}/logs"

# Repo root (for the pertoken overlay's import of nemo_rl sources when needed).
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="/lustre/fsw:/lustre/fsw" \
     --no-container-mount-home \
     bash -c "
set -euo pipefail
export HF_HOME='${HF_HOME}'
export MODEL_BF16='${MODEL_BF16}'
export MODEL_NVFP4='${MODEL_NVFP4}'
export MODEL_SMALL='${MODEL_SMALL}'
export RESULTS_DIR='${RESULTS_DIR}'
export PYTHONPATH='${SCRIPT_DIR}:${REPO_ROOT}'
export VLLM_LOGGING_LEVEL=\"\${VLLM_LOGGING_LEVEL:-INFO}\"

python3 -c 'import vllm; print(\"vllm\", vllm.__version__, getattr(vllm, \"__commit__\", \"?\"))'

python3 -m pip install -q pytest 2>/dev/null || true
cd '${SCRIPT_DIR}'

# One pytest process per check: engines leak KV-cache GPU memory across
# in-process engine rebuilds, so hard process isolation is the only reliable
# teardown. Failures do not stop later checks.
# Reload checks (check_c_*) run LAST: a reload crash can leave the GPU with
# dying engine procs that poison the next engine init.
CHECKS=\"\${RUN_CHECKS_LIST:-check_a_smoke_small check_a_smoke_qwen30b check_a_tp2 check_b1 check_b2 check_d1 check_d2 check_d3 check_d4 check_d5 check_c_reload_pertoken_small check_c_reload_hybrid}\"
wait_gpu_free() {
    for _ in \$(seq 1 60); do
        free_mb=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
        [ \"\${free_mb}\" -gt 150000 ] && return 0
        sleep 5
    done
    echo \"WARN: GPU0 still busy after 5min (free=\${free_mb}MB)\"
}
overall=0
for chk in \${CHECKS}; do
    wait_gpu_free
    echo \"================ RUNNING \${chk} ================\"
    python3 -m pytest -v -s -o addopts= --durations=5 test_nvfp4_pertoken.py \
        -k \"\${chk}\" --junitxml=\"${RESULTS_DIR}/junit-\${chk}.xml\" 2>&1 \
        || { echo \"CHECK_FAILED \${chk}\"; overall=1; }
done
echo \"================ SUITE DONE (overall=\${overall}) ================\"
exit \${overall}
"
