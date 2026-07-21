#!/bin/bash
#SBATCH --job-name=general_sa-nemo_rl.qwen3-30ba3b-dapo512-20k
#SBATCH --account=general_sa
#SBATCH --partition=batch,tcpo,36x2-a01r,a02grace
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --output=logs/dapo512-20k-%j.out
#SBATCH --error=logs/dapo512-20k-%j.err

# =============================================================================
# Qwen3-30B-A3B-Base DAPO-512, 20k response, GB200 8n x 4 GPU (tp2/ep8, vllm tp4).
# BF16 baseline vs NVFP4 per-token (MLP-only real-quant training + per-token W4A4
# rollout). Long run: 800 steps span multiple 5h allocations — a plain resubmit
# auto-resumes from the latest checkpoint and rejoins the pinned W&B run
# (checkpoint_must_save_by=4:15 forces a clean save before the 5h wall).
#
# Knobs (sbatch --export=ALL,...):
#   PRECISION  = bf16 (default) | nvfp4
#   MAX_STEPS  = override grpo.max_num_steps (default: recipe = 800)
#   EXTRA_ARGS = extra Hydra overrides (no commas)
# Usage (from repo root):
#   sbatch --export=ALL,GPUS_PER_NODE=4,PRECISION=bf16  research/vllm-nvfp4-pertoken/run_dapo_longrun.sh
#   sbatch --export=ALL,GPUS_PER_NODE=4,PRECISION=nvfp4 research/vllm-nvfp4-pertoken/run_dapo_longrun.sh
# Secrets (WANDB_API_KEY etc.) come from research/vllm-nvfp4-pertoken/.env (gitignored).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
NUM_NODES="${SLURM_NNODES:-8}"

ENV_FILE="${SCRIPT_DIR}/research/vllm-nvfp4-pertoken/.env"
[[ -f "${ENV_FILE}" ]] || ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
fi
# NVFP4 needs the TE-937c4de image (v4); allow an override for the bf16 leg.
CONTAINER_IMAGE="${CONTAINER_IMAGE_OVERRIDE:-${CONTAINER_IMAGE:-}}"

export TORCH_CUDA_ARCH_LIST='10.0'
: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-30b-nvfp4}"

PRECISION="${PRECISION:-bf16}"
case "${PRECISION}" in
    bf16)  RECIPE="examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-bf16.yaml" ;;
    nvfp4) RECIPE="examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken.yaml" ;;
    *) echo "ERROR: PRECISION must be bf16 or nvfp4 (got '${PRECISION}')." >&2; exit 1 ;;
esac
# RECIPE_OVERRIDE: run an arbitrary recipe (e.g. a probe variant) without the
# fragile EXTRA_ARGS quoting. Its basename drives RUN_NAME -> distinct dirs.
RECIPE="${RECIPE_OVERRIDE:-${RECIPE}}"
RUN_NAME="$(basename "${RECIPE}" .yaml)"
MAX_STEPS="${MAX_STEPS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CKPT_DIR="${CKPT_DIR:-${RL_DIR}/results/${RUN_NAME}}"
LOG_DIR_EXP="${LOG_DIR_EXP:-${RL_DIR}/logs/${RUN_NAME}}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR_EXP}" logs

# W&B run-id pin: a resubmit rejoins the same run so the 800-step curve is
# continuous across allocations. First submission mints an id from the name.
WANDB_RUN_ID_PIN="${CKPT_DIR}/.wandb_run_id"
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    if [[ -f "${WANDB_RUN_ID_PIN}" ]]; then
        WANDB_RUN_ID="$(head -n1 "${WANDB_RUN_ID_PIN}" | tr -d '[:space:]')"
    else
        WANDB_RUN_ID="$(printf '%s-%s' "${WANDB_PROJECT}" "${RUN_NAME}" | md5sum | cut -c1-32)"
    fi
fi
printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_PIN}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

GRPO_ARGS="--config ${RECIPE} \
    cluster.num_nodes=${NUM_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} \
    checkpointing.checkpoint_dir=${CKPT_DIR} \
    logger.log_dir=${LOG_DIR_EXP} \
    logger.wandb.project=${WANDB_PROJECT} \
    logger.wandb.name=${RUN_NAME}"
[[ -n "${MAX_STEPS}" ]] && GRPO_ARGS="${GRPO_ARGS} grpo.max_num_steps=${MAX_STEPS}"
[[ -n "${EXTRA_ARGS}" ]] && GRPO_ARGS="${GRPO_ARGS} ${EXTRA_ARGS}"

echo "=============================================="
echo "  Precision:  ${PRECISION}"
echo "  Recipe:     ${RECIPE}"
echo "  Run name:   ${RUN_NAME}"
echo "  Nodes/GPUs: ${NUM_NODES} x ${GPUS_PER_NODE}"
echo "  Container:  ${CONTAINER_IMAGE}"
echo "  Ckpt dir:   ${CKPT_DIR}"
echo "  W&B:        ${WANDB_PROJECT}/${RUN_NAME} (id=${WANDB_RUN_ID}, resume=${WANDB_RESUME})"
echo "  Max steps:  ${MAX_STEPS:-recipe default (800)}"
echo "  Job ID:     ${SLURM_JOB_ID:-interactive}"
echo "=============================================="

cd "${RL_DIR}"

export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl"
export GPUS_PER_NODE
export NEMO_RL_VENV_DIR="/opt/ray_venvs"

export COMMAND="\
    export NEMO_RL_VENV_DIR=/opt/ray_venvs && \
    export PYTHONUNBUFFERED=1 && \
    export UV_HTTP_TIMEOUT=900 && \
    export HF_HOME=${HF_HOME} && \
    export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
    export HF_TOKEN=${HF_TOKEN} && \
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_RUN_ID=${WANDB_RUN_ID} && \
    export WANDB_RESUME=${WANDB_RESUME} && \
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    export NRL_INSTALL_FA3=0 && \
    cd /opt/nemo-rl && \
    uv run --no-sync examples/run_grpo.py ${GRPO_ARGS}"

source ray.sub

echo "Job completed at: $(date)"
