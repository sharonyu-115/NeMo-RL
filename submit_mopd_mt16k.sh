#!/bin/bash
# v3 of the multi-teacher MOPD experiment: 16k generation budget, lr 1e-6,
# 300 steps as a chain of checkpoint-resuming 4h jobs (requires the
# replay-buffer watermark clamp fix on this branch), DAPO-format AIME24
# avg@8 math val, wandb entity nv-welcome.
#
# Usage: bash submit_mopd_mt16k.sh <arm:1|2|3> [num_chained_jobs] [extra overrides...]
set -eou pipefail

ARM=${1:?usage: submit_mopd_mt16k.sh <1|2|3> [num_jobs] [overrides...]}
NUM_JOBS=${2:-3}
shift $(( $# >= 2 ? 2 : 1 ))

REPO=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
USER_FS1=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy
USER_FSW=/lustre/fsw/portfolios/coreai/users/shuangy

set -a; source ~/.env; set +a
# v3 runs log to the nv-welcome entity with its own key.
export WANDB_API_KEY=${WANDB_API_KEY_NVWELCOME:?WANDB_API_KEY_NVWELCOME missing from ~/.env}

export HF_HOME=${USER_FS1}/src/NeMo-RL/hf
export HF_DATASETS_CACHE=${USER_FS1}/src/NeMo-RL/hf_datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

THINKING=Qwen/Qwen3-4B-Thinking-2507
GENERALIST=Qwen/Qwen3-4B
case $ARM in
  1) MATH_T=$THINKING;   IF_T=$THINKING;   NODES=3 ;;
  2) MATH_T=$GENERALIST; IF_T=$GENERALIST; NODES=3 ;;
  3) MATH_T=$THINKING;   IF_T=$GENERALIST; NODES=4 ;;
  *) echo "invalid arm $ARM"; exit 1 ;;
esac

RUN_NAME=mopd-mt16k-arm${ARM}

cd ${REPO}
DEP=""
for i in $(seq 1 ${NUM_JOBS}); do
  JOB_ID=$(
  COMMAND="uv run examples/nemo_gym/run_grpo_nemo_gym.py \
      --config examples/configs/recipes/llm/mopd-mt-qwen3-1.7b.yaml \
      on_policy_distillation.teacher_model_by_agent_name.default_teacher=${MATH_T} \
      on_policy_distillation.teacher_model_by_agent_name.instruction_following_simple_agent=${IF_T} \
      cluster.num_nodes=${NODES} \
      policy.generation.max_new_tokens=16384 \
      policy.megatron_cfg.optimizer.lr=1e-6 \
      policy.megatron_cfg.optimizer.min_lr=1e-6 \
      grpo.max_num_steps=300 \
      grpo.val_period=50 \
      data.train.data_path=${HF_HOME}/mopd_mt_data_v3/train-split.jsonl \
      data.validation.data_path=${HF_HOME}/mopd_mt_data_v3/val-split.jsonl \
      logger.wandb_enabled=True \
      ++logger.wandb.entity=nv-welcome \
      logger.wandb.name=${RUN_NAME} \
      logger.log_dir=${REPO}/results/${RUN_NAME}/logs \
      logger.monitor_gpus=True \
      checkpointing.checkpoint_dir=${REPO}/results/${RUN_NAME}/ckpts \
      $*" \
  SETUP_COMMAND="rm -rf /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym" \
  CONTAINER=${USER_FS1}/images/nemo-rl-mopd-main-2026-07-15.sqsh \
  MOUNTS="${USER_FS1}:${USER_FS1},${USER_FSW}:${USER_FSW}" \
  sbatch --parsable ${DEP} \
      --account=coreai_dlalgo_nemorl \
      --partition=batch \
      --nodes=${NODES} \
      --gres=gpu:8 \
      --time=4:00:00 \
      --job-name=${RUN_NAME} \
      ray.sub
  )
  echo "submitted ${RUN_NAME} job ${i}/${NUM_JOBS}: ${JOB_ID}"
  DEP="--dependency=afterany:${JOB_ID}"
done
