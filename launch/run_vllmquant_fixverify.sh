#!/bin/bash
#SBATCH --job-name=general_sa-nemo_rl.vllmquant-fixverify
#SBATCH --account=general_sa
#SBATCH --partition=batch,tcpo,36x2-a01r,a02grace
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --output=logs/vllmquant-fixverify-%j.out
#SBATCH --error=logs/vllmquant-fixverify-%j.err

# =============================================================================
# Post-fix regression check for shuangy/nvfp4-vllm-side-quant, started fresh
# from scratch entirely on commit d498d6d29 ("clone passthrough tensors
# during NVFP4 per-token layerwise reload") -- unlike run_vllmquant_verify.sh,
# whose leg 1-3 predate that fix. Checks whether the gen_kl_error spikes seen
# at steps 42/78 in that earlier run are gone under the fixed code.
#
# Single leg only (no chain keeper) -- checkpoint_must_save_by=4:15 in the
# recipe still forces a clean save before the 5h wall, so this leg is
# resumable by hand later if desired, but nothing auto-resubmits it.
# =============================================================================
set -euo pipefail

RL_DIR=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-vllmquant
CONTAINER_IMAGE=/lustre/fsw/general_sa/shuangy/images/nemo-rl-pr3566-smoke-2026-08-16-v2.sqsh
HF_HOME=/lustre/fsw/general_sa/shuangy/hf
RECIPE="${RECIPE_OVERRIDE:-examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-r3-rnquant-noolf-vllmquant-fixverify.yaml}"
WANDB_PROJECT=qwen3-30b-nvfp4
RUN_NAME="$(basename "${RECIPE}" .yaml)"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${SLURM_NNODES:-8}"

SECRETS=/lustre/fsw/general_sa/shuangy/tmp/pr3566-longrun/secrets.env
[[ -f "${SECRETS}" ]] || { echo "ERROR: ${SECRETS} not found." >&2; exit 1; }
set -a; source "${SECRETS}"; set +a
: "${WANDB_API_KEY:?Set WANDB_API_KEY in ${SECRETS}}"
HF_TOKEN="${HF_TOKEN:-}"

CKPT_DIR="${RL_DIR}/results/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" logs

# W&B run-id pin: a resubmit rejoins the same run so the curve is continuous
# across legs/allocations. First submission mints an id from the run name.
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

echo "=============================================="
echo "  Recipe:     ${RECIPE}"
echo "  Run name:   ${RUN_NAME}"
echo "  Nodes/GPUs: ${NUM_NODES} x ${GPUS_PER_NODE}"
echo "  Container:  ${CONTAINER_IMAGE}"
echo "  Ckpt dir:   ${CKPT_DIR}"
echo "  W&B:        nv-welcome/${WANDB_PROJECT}/${RUN_NAME} (id=${WANDB_RUN_ID}, resume=${WANDB_RESUME})"
echo "  Job ID:     ${SLURM_JOB_ID:-interactive}"
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
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_RUN_ID=${WANDB_RUN_ID} && \
    export WANDB_RESUME=${WANDB_RESUME} && \
    export NVTE_WITH_NCCL_EP=0 && \
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    export NRL_INSTALL_FA3=0 && \
    uv run examples/run_grpo.py --config ${RECIPE}"

source ray.sub

echo "Job completed at: $(date)"
