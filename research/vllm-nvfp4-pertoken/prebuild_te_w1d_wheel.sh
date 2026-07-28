#!/bin/bash
# Step 1 of the per-tensor-1D weight-leg bake: regenerate uv.lock for the re-pinned
# TE (sharonyu-115 fork @25e1fda6, per-tensor-1D weight patch on top of cael-ling
# @690ffea4) and compile the TE wheel DIRECTLY into the persistent lustre uv cache.
# Once cached, the actual bake (env_refresh_te_w1d.sh) reuses the wheel and is fast.
# No --container-save needed: UV_CACHE_DIR is on lustre and persists across jobs.
#
# Same two blockers as the 690ffea prebuild, both fatal if forgotten:
#   * NVTE_WITH_NCCL_EP=0    -- nccl_ep.cc needs NCCL GIN symbols this container lacks
#   * srun --cpus-per-task=128 -- without it the task cgroup gets nproc=1 and the
#                                 compile crawls for hours before failing
set -euo pipefail

REPO=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd
cd "$REPO"

export UV_CACHE_DIR=/lustre/fsw/general_sa/shuangy/uv_cache_te_bwd
export UV_HTTP_TIMEOUT=900
export TORCH_CUDA_ARCH_LIST='10.0'
export MAX_JOBS=64
export NVTE_BUILD_THREADS_PER_JOB=1
export NVTE_WITH_NCCL_EP=0
export CMAKE_BUILD_PARALLEL_LEVEL=64
export VERBOSE=1
export NVTE_VERBOSE=1

echo "=== nproc visible: $(nproc) ; MAX_JOBS=$MAX_JOBS ==="
grep -n "TransformerEngine.git@" pyproject.toml

echo "=== regenerating uv.lock for the re-pinned TE ==="
date
time uv lock --no-progress 2>&1
echo "--- lock now points at: ---"
grep -c "sharonyu-115/TransformerEngine" uv.lock
grep -c "cael-ling/TransformerEngine" uv.lock || true

echo "=== building mcore extra (TE @25e1fda6 from source) into $UV_CACHE_DIR ==="
date
time uv sync --extra mcore --locked --no-progress 2>&1
date

echo "=== is the patched TE wheel now in the cache? ==="
python3 - <<'EOF'
import glob, os
c = os.environ["UV_CACHE_DIR"]
hits = glob.glob(f"{c}/**/transformer_engine*.whl", recursive=True)
print("TE wheels cached:", len(hits))
for h in sorted(hits):
    print("  ", h)
EOF
echo "=== TE prebuild done ==="
