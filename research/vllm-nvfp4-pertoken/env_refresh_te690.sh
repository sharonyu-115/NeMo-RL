#!/bin/bash
# P1 gate-2 bake: rebuild /opt/ray_venvs from the te-bwd worktree's repinned
# pyproject+uv.lock (TransformerEngine -> cael-ling@690ffea, PR #3045) by running
# one tiny megatron GRPO step (compiles TE 690ffea into the mcore worker venv),
# then VERIFY the built TE actually carries the per-token BACKWARD switch.
# Run under `srun --container-save` (see env-refresh skill). 1 node / 4 GPUs.
set -e

REPO=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd
cd "$REPO"

export NRL_FORCE_REBUILD_VENVS=true
export WANDB_MODE=disabled
# Reuse the te_bwd uv cache — it already holds the 690ffea TE source from the
# passing gate-1 `uv lock`, so the mcore venv build resolves offline-fast.
export UV_CACHE_DIR=/lustre/fsw/general_sa/shuangy/uv_cache_te_bwd
export UV_HTTP_TIMEOUT=900
export HF_HOME=/lustre/fsw/general_sa/shuangy/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST='10.0'
# TE @690ffea builds from source (git ref). With --exclusive the node exposes
# 144 cores and the TE/Megatron builds default to ONE compiler job PER CORE ->
# memory thrash that stalled the compile past a 4h limit. Pin to the project's
# proven value (docker/Dockerfile.ngc_pytorch uses MAX_JOBS=32).
export MAX_JOBS=32
export NVTE_BUILD_THREADS_PER_JOB=2
# TE @690ffea's NCCL EP submodule fails to compile (needs newer NCCL GIN symbols
# than this container ships) and is irrelevant to NVFP4 GEMM precision. Disable
# it. Harmless if TE is already cached (built without EP); required if it rebuilds.
export NVTE_WITH_NCCL_EP=0

echo "=== gate-2 bake: rebuilding Ray worker venvs on TE 690ffea (megatron + vllm) ==="
LOG=/tmp/env_refresh_te690.log
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
        echo "=== First training step completed (venvs rebuilt on TE 690ffea) ==="
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

echo "=== Verifying venv versions + PR #3045 per-token BACKWARD switch ==="
MCORE_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
for V in "$MCORE_VENV" "$VLLM_VENV"; do
    [ -d "$V" ] && echo "--- $(basename "$V") present ---" || { echo "MISSING VENV: $V"; exit 1; }
done

# Critical gate: prove the freshly built TE is 690ffea AND that its NVFP4 recipe
# exposes the per-token backward knobs from PR #3045. If these are absent the
# bake produced a stale/wrong TE and the whole campaign is blocked.
"$MCORE_VENV/bin/python" - <<'EOF'
import os, sys, transformer_engine
te_dir = os.path.dirname(transformer_engine.__file__)
print("TE version:", transformer_engine.__version__)
print("TE path:", te_dir)

# Old per-token FORWARD switch must still exist (fwd not broken by the bump).
fwd = os.popen(f"grep -rl NVTE_NVFP4_ROW_SCALED_ACTIVATION {te_dir} 2>/dev/null | head -1").read().strip()
# NEW: PR #3045 per-token backward + opt-in knobs.
per_token = os.popen(f"grep -rl NVTE_NVFP4_PER_TOKEN {te_dir} 2>/dev/null | head -1").read().strip()
bwd_override = os.popen(f"grep -rl NVTE_BACKWARD_OVERRIDE {te_dir} 2>/dev/null | head -1").read().strip()

print("NVTE_NVFP4_ROW_SCALED_ACTIVATION present:", bool(fwd), "->", fwd)
print("NVTE_NVFP4_PER_TOKEN present:            ", bool(per_token), "->", per_token)
print("NVTE_BACKWARD_OVERRIDE present:          ", bool(bwd_override), "->", bwd_override)

# Confirm the recipe classes are importable and per-token subclass exists.
try:
    from transformer_engine.common.recipe import NVFP4BlockScaling
    print("NVFP4BlockScaling import: OK")
    try:
        from transformer_engine.common.recipe import NVFP4PerTokenBlockScaling
        print("NVFP4PerTokenBlockScaling import: OK")
    except Exception as e:
        print("NVFP4PerTokenBlockScaling import: MISSING ->", e)
except Exception as e:
    print("NVFP4BlockScaling import FAILED ->", e)

ok = bool(per_token) and bool(fwd) and bool(bwd_override)
print("PR #3045 backward switch present:", ok)
sys.exit(0 if ok else 3)
EOF
TE_CHECK=$?

# --- D1 CRUX: does PLAIN NVFP4BlockScaling (what Megatron builds) honor the
# per-token BACKWARD switch via env vars alone? Construct it in fresh processes
# under (a) today's env and (b) the enable-switch env, then diff the fields.
# If they differ, env-var-only wiring (design D1) is valid; if identical, the
# switch requires the NVFP4PerTokenBlockScaling subclass -> D1 is WRONG.
cat > /tmp/te_recipe_dump.py <<'EOF'
import dataclasses, json, os
from transformer_engine.common.recipe import NVFP4BlockScaling
r = NVFP4BlockScaling()
def field_view(obj):
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        out[f.name] = repr(v)
    return out
print(json.dumps({"env": {k: os.environ.get(k) for k in
    ("NVTE_NVFP4_PER_TOKEN","NVTE_BACKWARD_OVERRIDE","NVTE_NVFP4_ROW_SCALED_ACTIVATION")},
    "fields": field_view(r)}, indent=2, sort_keys=True))
EOF
echo "--- D1 (a) baseline: NVTE_BACKWARD_OVERRIDE=dequantized (today) ---"
env -u NVTE_NVFP4_PER_TOKEN NVTE_BACKWARD_OVERRIDE=dequantized \
    "$MCORE_VENV/bin/python" /tmp/te_recipe_dump.py > /tmp/te_recipe_a.json || true
cat /tmp/te_recipe_a.json
echo "--- D1 (b) switch: NVTE_NVFP4_PER_TOKEN=1, NVTE_BACKWARD_OVERRIDE unset ---"
env -u NVTE_BACKWARD_OVERRIDE NVTE_NVFP4_PER_TOKEN=1 \
    "$MCORE_VENV/bin/python" /tmp/te_recipe_dump.py > /tmp/te_recipe_b.json || true
cat /tmp/te_recipe_b.json
echo "--- D1 diff (empty diff => plain NVFP4BlockScaling IGNORES the switch => D1 WRONG) ---"
diff <(python3 -c "import json;print(json.load(open('/tmp/te_recipe_a.json'))['fields'])" 2>/dev/null) \
     <(python3 -c "import json;print(json.load(open('/tmp/te_recipe_b.json'))['fields'])" 2>/dev/null) \
     || echo "(fields differ above — switch is honored by the plain class)"

"$VLLM_VENV/bin/python" - <<'EOF'
import vllm
print("vllm:", vllm.__version__)
import vllm.model_executor.layers.quantization.online.nvfp4  # noqa: F401
print("online.nvfp4: OK")
EOF

if [ "$TE_CHECK" -ne 0 ]; then
    echo "=== FAIL: built TE lacks PR #3045 per-token backward switch — NOT a valid probe image ==="
    exit "$TE_CHECK"
fi

echo "=== gate-2 bake OK; container saves on exit ==="
