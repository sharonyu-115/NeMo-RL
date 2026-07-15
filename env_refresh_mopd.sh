#!/bin/bash
# Refresh the container env for the MOPD worktree and bake a matching image.
#
# Runs a 1-node megatron GRPO step to force-build the base env and all Ray
# worker venvs (serialized via NeMo-RL's env builders), verifies imports,
# regenerates the container fingerprint, then exits so sbatch --container-save
# snapshots the result. Fixes the drift-corruption failure mode where raylet
# worker spawns concurrently `uv sync` a stale /opt/nemo_rl_venv in place
# (observed: transformers 'unknown location', missing ray._private.node).
set -e

REPO=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/src/NeMo-RL/nemo-rl-mopd
cd "$REPO"

# Secrets from ~/.env — never hardcode.
set -a; source ~/.env; set +a

export NRL_FORCE_REBUILD_VENVS=true
export WANDB_MODE=disabled
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/src/NeMo-RL/hf
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/src/NeMo-RL/hf_datasets
export UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/shuangy/uv_cache
export UV_HTTP_TIMEOUT=600
mkdir -p "$UV_CACHE_DIR"

echo "=== MOPD env refresh: REPO=$REPO ==="
echo "=== Start time: $(date) ==="

LOG=/tmp/env_refresh_mopd.log
rm -f "$LOG"
# Megatron backend + Qwen3-1.7B (already in HF cache) to build the venvs the
# MOPD recipe uses (megatron policy/teacher + vllm generation).
uv run --extra mcore python examples/run_grpo.py \
    policy.model_name=Qwen/Qwen3-1.7B \
    policy.megatron_cfg.enabled=true \
    policy.dtensor_cfg.enabled=false \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    grpo.max_num_steps=1 \
    checkpointing.enabled=false \
    logger.wandb_enabled=false \
    2>&1 | tee "$LOG" &
TRAIN_PID=$!

echo "=== Waiting for first training step (up to 90 min for venv builds) ==="
START_WAIT=$(date +%s)
while true; do
    if grep -q "Total step time" "$LOG" 2>/dev/null; then
        echo "=== First training step completed after $(( $(date +%s) - START_WAIT ))s ==="
        break
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "=== Training process exited unexpectedly; dumping log ==="
        tail -n 200 "$LOG"
        exit 1
    fi
    sleep 15
done

echo "=== Killing training process ==="
kill $TRAIN_PID 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f "run_grpo" 2>/dev/null || true
sleep 5

echo "=== Verifying package versions ==="
MEGATRON_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker

for V in "$MEGATRON_VENV" "$VLLM_VENV"; do
    if [ -d "$V" ]; then
        echo "--- $V ---"
        "$V/bin/python" -c "import torch; print(f'torch: {torch.__version__}')"
        "$V/bin/python" -c "import transformers; print(f'transformers: {transformers.__version__}')"
    else
        echo "WARNING: venv not found at $V"
    fi
done
"$VLLM_VENV/bin/python" -c "import vllm; print(f'vllm: {vllm.__version__}')"
"$VLLM_VENV/bin/python" -c "import ray; print(f'ray: {ray.__version__}')"

# Stamp the fingerprint so future runs of this checkout skip the in-place
# drift sync (the root cause of the venv corruption).
echo "=== Updating container fingerprint ==="
uv run python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
cat /opt/nemo_rl_container_fingerprint

echo "=== Environment refresh complete: $(date) ==="
echo "=== Container will be saved on exit via --container-save ==="
