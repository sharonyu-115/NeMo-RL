#!/bin/bash
# A3 venv bake: rebuild /opt/ray_venvs from the repinned pyproject+uv.lock by
# running one tiny megatron GRPO step (builds mcore + vllm worker venvs), then
# verify versions. Run under srun --container-save (see PROGRESS.md).
set -e

REPO=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl
cd "$REPO"

export NRL_FORCE_REBUILD_VENVS=true
export WANDB_MODE=disabled
export UV_CACHE_DIR=/lustre/fsw/general_sa/shuangy/tmp/uv-cache
export UV_HTTP_TIMEOUT=900
export HF_HOME=/lustre/fsw/general_sa/shuangy/hf
# Model + dataset are in the local HF cache; stay offline for HF (pip still
# has network for the venv builds).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST='10.0'

echo "=== env refresh: rebuilding Ray worker venvs (megatron + vllm) ==="
LOG=/tmp/env_refresh.log
uv run python examples/run_grpo.py \
    --config examples/configs/grpo_math_1B_megatron.yaml \
    grpo.max_num_steps=1 \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=16 \
    cluster.gpus_per_node=4 \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    2>&1 | tee "$LOG" &
TRAIN_PID=$!

while true; do
    if grep -q "Total step time" "$LOG" 2>/dev/null; then
        echo "=== First training step completed ==="
        break
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "=== Training process exited before step 1 ==="
        tail -50 "$LOG"
        exit 1
    fi
    sleep 15
done

kill $TRAIN_PID 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f run_grpo 2>/dev/null || true
sleep 5

echo "=== Verifying venv versions ==="
MCORE_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
for V in "$MCORE_VENV" "$VLLM_VENV"; do
    [ -d "$V" ] && echo "--- $(basename "$V") ---" || { echo "MISSING VENV: $V"; exit 1; }
done
"$VLLM_VENV/bin/python" - <<'EOF'
import vllm, flashinfer
print("vllm:", vllm.__version__)
print("flashinfer:", flashinfer.__version__)
import vllm.model_executor.layers.quantization.online.nvfp4  # noqa: F401
print("online.nvfp4: OK")
EOF
"$MCORE_VENV/bin/python" - <<'EOF'
import os, transformer_engine
te_dir = os.path.dirname(transformer_engine.__file__)
print("TE:", transformer_engine.__version__)
hits = os.popen(f"grep -rl NVTE_NVFP4_ROW_SCALED_ACTIVATION {te_dir} | head -3").read().strip()
print("NVTE_NVFP4_ROW_SCALED_ACTIVATION present:", bool(hits))
EOF

echo "=== Environment refresh complete; container saves on exit ==="
