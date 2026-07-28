#!/bin/bash
# Bake the flashinfer cubin cache INTO the probe image.
#
# Why: flashinfer JIT-fetches trtllm-gen cubins from edge.urm.nvidia.com at vLLM
# warmup, into /root/.cache/flashinfer/cubins — which lives in the container
# overlay and is therefore thrown away every allocation. Each of the ~13
# allocations per 1500-step leg re-rolls that download, and on 2026-07-27 it
# failed 2 of 4 times ("Failed to get checksums.txt", plus 30s lock timeouts as
# the 4 workers/node pile onto the same lock). A failed fetch kills vLLM warmup
# and burns the whole allocation.
#
# A lustre FLASHINFER_CUBIN_DIR would also persist the cache, but it would put
# 32 processes on lustre file locks — and lock contention is half of what we
# already saw fail. Baking keeps the reads node-local.
#
# The staged set is not a guess: it is the exact list of cubins fetched by the
# two successful 8n4g runs (jobs 2455672 + 2456095), extracted from their driver
# logs, plus the per-bucket checksums.txt that the loader validates against.
# Downloaded from the LOGIN node, where the endpoint is reliable, so this bake
# never depends on compute-node egress.
#
# Run under `srun --container-save=<NEW image>`.
set -euo pipefail

STAGE=/lustre/fsw/general_sa/shuangy/flashinfer_cubins_stage
DEST=/root/.cache/flashinfer/cubins

echo "=== staging flashinfer cubins into the image ==="
[ -d "$STAGE" ] || { echo "FAIL: stage dir $STAGE missing"; exit 1; }
staged=$(find "$STAGE" -type f | wc -l)
echo "staged files on lustre: $staged"
[ "$staged" -ge 40 ] || { echo "FAIL: expected >=40 staged files, found $staged"; exit 1; }

mkdir -p "$DEST"
cp -r "$STAGE"/. "$DEST"/
baked=$(find "$DEST" -type f | wc -l)
echo "files now in $DEST: $baked"
[ "$baked" -ge "$staged" ] || { echo "FAIL: copy incomplete ($baked < $staged)"; exit 1; }
du -sh "$DEST"

# Gate: make flashinfer itself resolve the file whose absence killed two runs,
# with downloads DISABLED. If the loader reads it from the baked cache, a
# network-less allocation can get through warmup.
echo "=== gate: flashinfer resolves the cached cubin with downloads disabled ==="
VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_worker.NvFp4PerTokenGenerationWorker
[ -d "$VLLM_VENV" ] || VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
[ -d "$VLLM_VENV" ] || { echo "FAIL: no vllm worker venv found under /opt/ray_venvs"; exit 1; }
echo "using venv: $VLLM_VENV"

FLASHINFER_NO_DOWNLOAD=1 "$VLLM_VENV/bin/python" - <<'EOF'
import sys

from flashinfer.jit.env import FLASHINFER_CUBIN_DIR
from flashinfer.jit.cubin_loader import get_cubin

print("FLASHINFER_CUBIN_DIR:", FLASHINFER_CUBIN_DIR)

BUCKETS = [
    # The bucket whose checksums.txt fetch killed jobs 2455982 and 2456030.
    "481dce07c89a216cbfd18cf39de49a82d40739a8/batched_gemm-dd6d23e-721ae60",
    "158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen",
]

failures = []
checked = 0
for bucket in BUCKETS:
    sums = FLASHINFER_CUBIN_DIR / bucket / "checksums.txt"
    if not sums.is_file():
        failures.append(f"{bucket}/checksums.txt missing from the baked cache")
        continue

    # checksums.txt is "<sha256>  <filename>" per line. Runtime looks up the
    # expected hash there and passes it to get_cubin, which returns b"" on a
    # mismatch and then (offline) raises. So probing with the REAL hash is the
    # only way to prove the cached file would actually satisfy a live worker --
    # passing a dummy hash always misses, regardless of the cache.
    expected = {}
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1]] = parts[0]

    # Probe every cubin we staged for this bucket, not just one: a partial or
    # truncated copy would otherwise pass on the single file we happened to pick.
    staged = sorted(p for p in (FLASHINFER_CUBIN_DIR / bucket).glob("*.cubin"))
    if not staged:
        failures.append(f"{bucket}: no .cubin files in the baked cache")
        continue

    for path in staged:
        sha = expected.get(path.name)
        if sha is None:
            failures.append(f"{bucket}/{path.name}: absent from checksums.txt")
            continue
        try:
            data = get_cubin(f"{bucket}/{path.name}", sha)
        except Exception as e:  # noqa: BLE001 - report whatever the loader raised
            failures.append(f"{bucket}/{path.name}: {type(e).__name__}: {e}")
            continue
        if not data:
            failures.append(f"{bucket}/{path.name}: loader returned no data (sha mismatch)")
            continue
        checked += 1

    print(f"{bucket}: {len(staged)} cubins probed")

n = sum(1 for p in FLASHINFER_CUBIN_DIR.rglob("*") if p.is_file())
print(f"total cubin files in image: {n}")
print(f"cubins served offline by the loader: {checked}")

if failures:
    print("=== FAILED CHECKS ===")
    for f in failures[:20]:
        print("  -", f)
    sys.exit(3)
if checked < 39:
    print(f"FAIL: only {checked} cubins verified, expected >= 39")
    sys.exit(3)
print("all staged cubins load offline with their published checksums")
sys.exit(0)
EOF

echo "=== flashinfer cubin bake OK; container saves on exit ==="
