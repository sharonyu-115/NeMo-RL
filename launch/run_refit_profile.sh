#!/bin/bash
#SBATCH --job-name=general_sa-nemo_rl.refit-profile
#SBATCH --account=general_sa
#SBATCH --partition=batch,tcpo,36x2-a01r,a02grace
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/refit-profile-%j.out
#SBATCH --error=logs/refit-profile-%j.err

# =============================================================================
# Phase 0 of the NVFP4 per-token refit optimization: MEASURE ONLY.
#
# Runs the 4n4g quick recipe for 3 steps to collect
#   1. the per-refit wall-clock breakdown line emitted by
#      NvFp4PerTokenWorkerExtension (quantize / load / finalize / other), and
#   2. a cProfile call graph of one steady-state refit.
#
# Refit index 1, not 0: the first refit pays cold-start costs (FlashInfer JIT,
# first-touch allocations, venv warm-up) that the steady state does not, so
# profiling it would misattribute the recurring cost.
#
# No optimization is applied by this job. Checkpointing and W&B are off --
# nothing here should land in the qwen3-30b-nvfp4 project alongside the real
# convergence arms.
# =============================================================================
set -euo pipefail

RL_DIR=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-vllmquant
CONTAINER_IMAGE=/lustre/fsw/general_sa/shuangy/images/nemo-rl-pr3566-smoke-2026-08-16-v2.sqsh
HF_HOME=/lustre/fsw/general_sa/shuangy/hf
RECIPE="${RECIPE_OVERRIDE:-examples/configs/recipes/llm/grpo-qwen3-30ba3b-4n4g-megatron-te-nvfp4-pertoken-quick.yaml}"
RUN_NAME="$(basename "${RECIPE}" .yaml)"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${SLURM_NNODES:-4}"
MAX_STEPS="${MAX_STEPS:-3}"

# 0-based refit index to profile, and where the .pstats lands. Both are read at
# import time by nemo_rl/models/generation/vllm/vllm_backend.py; the driver
# environment reaches the vLLM workers because virtual_cluster.py forwards all
# of os.environ into the Ray runtime_env.
REFIT_PROFILE_STEP="${REFIT_PROFILE_STEP:-1}"
PROFILE_DIR="${RL_DIR}/session/20260821_nvfp4-refit-optimization/profiles/${SLURM_JOB_ID:-interactive}"

SECRETS=/lustre/fsw/general_sa/shuangy/tmp/pr3566-longrun/secrets.env
[[ -f "${SECRETS}" ]] || { echo "ERROR: ${SECRETS} not found." >&2; exit 1; }
set -a; source "${SECRETS}"; set +a
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "${PROFILE_DIR}" logs

echo "=============================================="
echo "  Recipe:       ${RECIPE}"
echo "  Nodes/GPUs:   ${NUM_NODES} x ${GPUS_PER_NODE}"
echo "  Steps:        ${MAX_STEPS}"
echo "  Container:    ${CONTAINER_IMAGE}"
echo "  Profile refit ${REFIT_PROFILE_STEP} -> ${PROFILE_DIR}"
echo "  Job ID:       ${SLURM_JOB_ID:-interactive}"
echo "=============================================="

cd "${RL_DIR}"

export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre"
export GPUS_PER_NODE
export NEMO_RL_VENV_DIR="/opt/ray_venvs"

export COMMAND="\
    cd ${RL_DIR} && \
    export NEMO_RL_VENV_DIR=/opt/ray_venvs && \
    export PYTHONUNBUFFERED=1 && \
    export UV_HTTP_TIMEOUT=900 && \
    export HF_HOME=${HF_HOME} && \
    export TORCH_CUDA_ARCH_LIST='10.0' && \
    export HF_TOKEN=${HF_TOKEN} && \
    export NVTE_WITH_NCCL_EP=0 && \
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    export NRL_INSTALL_FA3=0 && \
    export NRL_REFIT_PROFILE_STEP=${REFIT_PROFILE_STEP} && \
    export NRL_REFIT_PROFILE_DIR=${PROFILE_DIR} && \
    uv run examples/run_grpo.py --config ${RECIPE} \
        grpo.max_num_steps=${MAX_STEPS} \
        checkpointing.enabled=False \
        logger.wandb_enabled=False \
        logger.tensorboard_enabled=False"

source ray.sub

echo "Job completed at: $(date)"
