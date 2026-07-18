#!/bin/bash
#SBATCH --job-name=nvfp4-pertoken-probe
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
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

python -c 'import vllm; print(\"vllm\", vllm.__version__, getattr(vllm, \"__commit__\", \"?\"))'

pip install -q pytest 2>/dev/null || true
cd '${SCRIPT_DIR}'
python -m pytest -v -s test_nvfp4_pertoken.py -k \"${RUN_CHECKS}\" \
    --junitxml=\"${RESULTS_DIR}/junit.xml\" 2>&1
"
