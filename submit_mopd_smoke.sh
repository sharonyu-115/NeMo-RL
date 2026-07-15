#!/bin/bash
# Launch the MOPD 5-step smoke (PR #2780 recipe) on 3 nodes via ray.sub.
#
# Prereqs (see tools/prepare_mopd_gym_data.py):
#   $HF_HOME/nanov3_data/{train,val}-split.jsonl and Qwen/Qwen3-1.7B in the HF cache.
#
# Usage: bash submit_mopd_smoke.sh [extra config overrides...]
set -eou pipefail

REPO=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
USER_FS1=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy
USER_FSW=/lustre/fsw/portfolios/coreai/users/shuangy

# Secrets (HF_TOKEN, WANDB_API_KEY) from ~/.env — never hardcode here.
set -a; source ~/.env; set +a

export HF_HOME=${USER_FS1}/src/NeMo-RL/hf
export HF_DATASETS_CACHE=${USER_FS1}/src/NeMo-RL/hf_datasets
# Assets are pre-staged; avoid HF-hub flakiness/quota at scale.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

RUN_NAME=mopd-smoke-$(date +%m%d-%H%M)

cd ${REPO}
# Build a complete project venv on lustre instead of incrementally syncing the
# container's baked env: the image is older than the checked-out main (pyproject/
# uv.lock/Gym drift), and an in-place sync of the stale env leaves broken
# packages (observed: transformers 'unknown location' ImportError).
COMMAND="export UV_PROJECT_ENVIRONMENT=${REPO}/.venv && uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml \
    logger.wandb_enabled=True \
    logger.wandb.project=mopd \
    logger.wandb.name=${RUN_NAME} \
    logger.tensorboard_enabled=True \
    logger.log_dir=${REPO}/results/${RUN_NAME}/logs \
    logger.monitor_gpus=True \
    checkpointing.enabled=False \
    $*" \
CONTAINER=${USER_FS1}/images/nemo-rl-fp4-fa4-venvs-sm90fix-2026-07-10.sqsh \
MOUNTS="${USER_FS1}:${USER_FS1},${USER_FSW}:${USER_FSW}" \
UV_CACHE_DIR_OVERRIDE=${USER_FS1}/.uv-cache-main \
sbatch \
    --account=coreai_dlalgo_nemorl \
    --partition=batch \
    --nodes=3 \
    --gres=gpu:8 \
    --time=4:00:00 \
    --job-name=mopd-smoke \
    ray.sub
