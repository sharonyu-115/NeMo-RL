#!/bin/bash
# Step 2 of the per-tensor-1D weight-leg bake: rebuild /opt/ray_venvs from the
# re-pinned pyproject+uv.lock (TE -> sharonyu-115@25e1fda6) by running one tiny
# megatron GRPO step, then VERIFY the built TE carries BOTH the PR #3045 per-token
# backward switch AND the new per-tensor-1D weight flag, including a real GPU cast
# check of all three weight legs.
#
# Run under `srun --container-save=<NEW image>` — do NOT overwrite
# nemo-rl-te690ffea-probe.sqsh; the fp4bwd-r3 chain is still running on it.
set -e

REPO=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd
cd "$REPO"

export NRL_FORCE_REBUILD_VENVS=true
export WANDB_MODE=disabled
export UV_CACHE_DIR=/lustre/fsw/general_sa/shuangy/uv_cache_te_bwd
export UV_HTTP_TIMEOUT=900
export HF_HOME=/lustre/fsw/general_sa/shuangy/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST='10.0'
export MAX_JOBS=32
export NVTE_BUILD_THREADS_PER_JOB=2
export NVTE_WITH_NCCL_EP=0

echo "=== w1d bake: rebuilding Ray worker venvs on TE 25e1fda6 (megatron + vllm) ==="
grep -n "TransformerEngine.git@" pyproject.toml
LOG=/tmp/env_refresh_te_w1d.log
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
        echo "=== First training step completed (venvs rebuilt on TE 25e1fda6) ==="
        break
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "=== Training process exited before step 1 ==="
        tail -80 "$LOG"
        exit 1
    fi
    sleep 15
done

kill $TRAIN_PID 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f run_grpo 2>/dev/null || true
sleep 5

echo "=== Verifying venvs + the per-tensor-1D weight flag ==="
MCORE_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
for V in "$MCORE_VENV" "$VLLM_VENV"; do
    [ -d "$V" ] && echo "--- $(basename "$V") present ---" || { echo "MISSING VENV: $V"; exit 1; }
done

# Gate A (static): the built TE must be our patched revision and must contain the
# new env var string in the compiled extension. If the bake silently reused a
# cached 690ffea wheel this fails here rather than 20 GPU-hours later.
"$MCORE_VENV/bin/python" - <<'EOF'
import os, sys, transformer_engine
te_dir = os.path.dirname(transformer_engine.__file__)
print("TE version:", transformer_engine.__version__)
print("TE path:", te_dir)

# PR #3045 knobs (must survive the re-pin).
per_token = os.popen(f"grep -rl NVTE_NVFP4_PER_TOKEN {te_dir} 2>/dev/null | head -1").read().strip()
bwd = os.popen(f"grep -rl NVTE_BACKWARD_OVERRIDE {te_dir} 2>/dev/null | head -1").read().strip()
# NEW: the per-tensor-1D weight flag, compiled into the .so (binary grep).
w1d = os.popen(
    f"grep -rl NVTE_NVFP4_PER_TOKEN_WEIGHT_PER_TENSOR_1D {te_dir} 2>/dev/null | head -1"
).read().strip()

print("NVTE_NVFP4_PER_TOKEN present:                    ", bool(per_token), "->", per_token)
print("NVTE_BACKWARD_OVERRIDE present:                  ", bool(bwd), "->", bwd)
print("NVTE_NVFP4_PER_TOKEN_WEIGHT_PER_TENSOR_1D present:", bool(w1d), "->", w1d)

ok = bool(per_token) and bool(bwd) and bool(w1d)
print("patched TE present:", ok)
if not ok:
    print("FAIL: built TE is not the patched revision (25e1fda6)")
sys.exit(0 if ok else 3)
EOF
TE_CHECK=$?

if [ "$TE_CHECK" -ne 0 ]; then
    echo "=== FAIL: built TE lacks the per-tensor-1D weight flag — NOT a valid probe image ==="
    exit "$TE_CHECK"
fi

# Gate B (runtime, GPU): all three weight legs behave as designed. The crux is
# that leg C (per-tensor 1D) is direction-DEPENDENT while leg B (2D) is not.
echo "=== Gate B: GPU cast check of the three weight legs ==="
"$MCORE_VENV/bin/python" research/vllm-nvfp4-pertoken/verify_w1d_weight_legs.py
LEG_CHECK=$?

if [ "$LEG_CHECK" -ne 0 ]; then
    echo "=== FAIL: weight-leg cast verification failed — NOT a valid probe image ==="
    exit "$LEG_CHECK"
fi

"$VLLM_VENV/bin/python" - <<'EOF'
import vllm
print("vllm:", vllm.__version__)
import vllm.model_executor.layers.quantization.online.nvfp4  # noqa: F401
print("online.nvfp4: OK")
EOF

echo "=== w1d bake OK; container saves on exit ==="
