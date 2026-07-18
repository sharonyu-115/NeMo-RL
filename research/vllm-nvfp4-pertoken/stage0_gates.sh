#!/bin/bash
# Stage-0 environment gates: run on a GB200 node inside the vLLM nightly
# container. Usage: ./stage0_gates.sh   (submits itself via srun)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; source "${SCRIPT_DIR}/.env"; set +a

srun --account=general_sa --partition=batch,tcpo,36x2-a01r,a02grace --nodes=1 --exclusive \
     --time=00:15:00 --job-name=general_sa-nemo_rl.nvfp4-stage0-gates \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="/lustre/fsw:/lustre/fsw" \
     --no-container-mount-home \
     bash -c '
set -euo pipefail
echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo "=== vLLM version/commit ==="
python3 -c "import vllm; print(vllm.__version__); print(getattr(vllm, \"__commit__\", \"no commit attr\"))"
echo "=== Gate: Nvfp4OnlineMoEMethod importable ==="
python3 -c "from vllm.model_executor.layers.quantization.online.nvfp4 import Nvfp4OnlineMoEMethod; print(\"OK:\", Nvfp4OnlineMoEMethod)"
echo "=== Gate: oracle per_token_activation kwarg ==="
python3 -c "
import inspect
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import make_nvfp4_moe_kernel
sig = inspect.signature(make_nvfp4_moe_kernel)
assert \"per_token_activation\" in sig.parameters, sig
print(\"OK:\", sig)
"
echo "=== Gate: flashinfer TRT-LLM fused MoE available ==="
python3 -c "
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
print(\"has_flashinfer_trtllm_fused_moe:\", has_flashinfer_trtllm_fused_moe())
"
echo "ALL STAGE-0 GATES PASSED"
'
