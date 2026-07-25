#!/bin/bash
# Pre-build the TransformerEngine @690ffea wheel DIRECTLY (no Ray _env_builder,
# no num_cpus=1 actor) into the PERSISTENT lustre uv cache, with explicit high
# parallelism + verbose ninja output. Once the wheel is cached here, the actual
# gate-2 bake (run_grpo + NRL_FORCE_REBUILD_VENVS) reuses it and finishes fast.
# No --container-save needed: UV_CACHE_DIR is on lustre and persists across jobs.
set -euo pipefail

REPO=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd
cd "$REPO"

export UV_CACHE_DIR=/lustre/fsw/general_sa/shuangy/uv_cache_te_bwd
export UV_HTTP_TIMEOUT=900
export TORCH_CUDA_ARCH_LIST='10.0'
# Bounded, high parallelism (node has 144 cores / ~940GB RAM). 64 parallel nvcc
# jobs ~= up to ~256GB peak — safe. Set explicitly here (this shell IS the build
# env, unlike the Ray actor path where MAX_JOBS propagation was unverified).
export MAX_JOBS=64
export NVTE_BUILD_THREADS_PER_JOB=1
# TE @690ffea's NCCL EP submodule (GPU-initiated networking / comm-overlap)
# needs newer NCCL "GIN"/device-comm symbols than this container ships, so
# nccl_ep.cc fails to compile (ncclGetPeerDevicePointer / ncclCommQueryProperties
# / NCCL_GIN_* undeclared). NCCL EP is irrelevant to NVFP4 GEMM precision — turn
# it off (TE setup.py honors this -> -DNVTE_WITH_NCCL_EP=OFF). Arch 10.0 >= 90 so
# it will NOT auto-skip; the env var must be explicit.
export NVTE_WITH_NCCL_EP=0
# Verbose so ninja "[X/Y]" progress + the real job concurrency are visible.
export CMAKE_BUILD_PARALLEL_LEVEL=64
export VERBOSE=1
export NVTE_VERBOSE=1

echo "=== nproc visible: $(nproc) ; MAX_JOBS=$MAX_JOBS ==="
echo "=== building mcore extra (TE @690ffea from source) into $UV_CACHE_DIR ==="
date
# Time the TE build. uv sync builds all mcore deps incl. transformer-engine and
# caches the built wheel keyed by (git url, rev, interpreter/platform).
time uv sync --extra mcore --locked --no-progress 2>&1
date

echo "=== is the TE @690ffea wheel now in the cache? ==="
find "$UV_CACHE_DIR" -iname "transformer_engine*.whl" 2>/dev/null | head
python3 - <<'EOF'
import glob, os
c = os.environ["UV_CACHE_DIR"]
hits = glob.glob(f"{c}/**/transformer_engine*.whl", recursive=True)
print("TE wheels cached:", len(hits))
for h in hits[:5]:
    print("  ", h)
EOF
echo "=== TE prebuild done ==="
