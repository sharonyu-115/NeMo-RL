#!/bin/bash
#SBATCH --job-name=general_sa-nemo_rl.nvfp4-pertoken-grpo
#SBATCH --account=general_sa
#SBATCH --partition=batch,tcpo,36x2-a01r,a02grace
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/nvfp4-pertoken-grpo-%j.out
#SBATCH --error=logs/nvfp4-pertoken-grpo-%j.err

# =============================================================================
# Per-token NVFP4 W4A4 rollout e2e on GB200/ptyche. Runs the test-suite driver
# (greps + metric gates included) inside the repinned container via ray.sub.
# No --gpus-per-node directive (no-GRES cluster) — GPUS_PER_NODE via --export.
#
# Knobs (sbatch --export, no commas in EXTRA_ARGS):
#   DRIVER     — driver script under tests/test_suites/llm/
#                (default grpo-qwen3-30ba3b-4n4g-megatron-nvfp4-pertoken.sh;
#                 use the -fp4train variant for the M2 leg)
#   MAX_STEPS  — forwarded to the driver (default: driver's own default)
#   EXTRA_ARGS — extra Hydra overrides appended to the driver call
# Usage (from repo root):
#   sbatch --export=GPUS_PER_NODE=4 research/vllm-nvfp4-pertoken/run_pertoken_grpo.sh
#   sbatch --export=GPUS_PER_NODE=4,MAX_STEPS=50 research/vllm-nvfp4-pertoken/run_pertoken_grpo.sh
# 1-node refit smoke:
#   sbatch --export=GPUS_PER_NODE=4,MAX_STEPS=1,EXTRA_ARGS='cluster.num_nodes=1 policy.megatron_cfg.expert_model_parallel_size=4' -N1 research/vllm-nvfp4-pertoken/run_pertoken_grpo.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}}"
NUM_NODES="${SLURM_NNODES:-4}"

ENV_FILE="${SCRIPT_DIR}/research/vllm-nvfp4-pertoken/.env"
[[ -f "${ENV_FILE}" ]] || ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
fi

export TORCH_CUDA_ARCH_LIST='10.0'
: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

DRIVER="${DRIVER:-grpo-qwen3-30ba3b-4n4g-megatron-nvfp4-pertoken.sh}"
MAX_STEPS="${MAX_STEPS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

DRIVER_ENV=""
[[ -n "${MAX_STEPS}" ]] && DRIVER_ENV="export MAX_STEPS=${MAX_STEPS} && "

echo "=============================================="
echo "  Driver:     ${DRIVER}"
echo "  Nodes:      ${NUM_NODES} x ${GPUS_PER_NODE} GPUs"
echo "  Container:  ${CONTAINER_IMAGE}"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "  Job ID:     ${SLURM_JOB_ID:-interactive}"
echo "=============================================="

cd "${RL_DIR}"
mkdir -p logs

export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl"
export GPUS_PER_NODE
# Baked venvs: use the container's /opt/ray_venvs (no rebuild).
export NEMO_RL_VENV_DIR="/opt/ray_venvs"

export COMMAND="\
    ${DRIVER_ENV}\
    export NEMO_RL_VENV_DIR=/opt/ray_venvs && \
    export PYTHONUNBUFFERED=1 && \
    export UV_HTTP_TIMEOUT=900 && \
    export HF_HOME=${HF_HOME} && \
    export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
    ${HF_TOKEN:+export HF_TOKEN=${HF_TOKEN} && }\
    ${WANDB_API_KEY:+export WANDB_API_KEY=${WANDB_API_KEY} && }\
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    export NRL_INSTALL_FA3=0 && \
    cd /opt/nemo-rl && \
    bash tests/test_suites/llm/${DRIVER} ${EXTRA_ARGS}"

source ray.sub

echo "Job completed at: $(date)"
