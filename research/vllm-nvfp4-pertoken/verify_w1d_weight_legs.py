#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the three NVFP4 per-token WEIGHT legs on a built TE (SM100 required).

Mirrors the pytest cases added to the TE fork
(tests/pytorch/nvfp4/test_nvfp4_per_token.py, section 11) but runs standalone so
it can execute inside a Ray worker venv that has no pytest.

Legs under test, all with per-token activations:
  A. per-token 1D        -- per-row outer amax, 16-element inner blocks (default)
  B. per-tensor 2D       -- scalar outer amax, 16x16 inner tiles (WEIGHT_2D=1)
  C. per-tensor 1D       -- scalar outer amax, 16-element inner blocks
                            (WEIGHT_2D=1 + WEIGHT_PER_TENSOR_1D=1, the new leg)

The discriminating check is direction-dependence: B must dequantize identically
rowwise and columnwise-transposed; C must not. If C matches B, the env var never
reached the cast and the build is not a valid probe image.

Exit 0 on success, 3 on a failed check.
"""

from __future__ import annotations

import os
import sys

import torch

import transformer_engine.pytorch as te  # noqa: F401  (dlopen order matters)
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

W_PER_TENSOR_1D_ENV = "NVTE_NVFP4_PER_TOKEN_WEIGHT_PER_TENSOR_1D"
BLOCK_K = 16
SHAPES = [(256, 256), (512, 1024), (1024, 1024)]


def make_quantizer(per_token: bool, per_token_weight_2d: bool, with_2d: bool) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=with_2d,
        per_token=per_token,
        per_token_weight_2d=per_token_weight_2d,
    )


def quantize(quantizer: NVFP4Quantizer, w: torch.Tensor):
    dst = quantizer.make_empty(w.shape, dtype=w.dtype, device=w.device, requires_grad=False)
    return quantizer.update_quantized(w, dst)


def dequant(data: torch.Tensor, scale_inv: torch.Tensor, outer_amax: torch.Tensor) -> torch.Tensor:
    """Decode FP4 with the kernel's arithmetic: q * s_dec * (6 / S_enc_row)."""
    q = data.view(torch.uint8)
    rows, half_cols = q.shape
    cols = half_cols * 2
    sf = scale_inv[:rows, : cols // BLOCK_K].view(torch.float8_e4m3fn).to(torch.float32)

    lo = (q & 0x0F).to(torch.int8)
    hi = ((q >> 4) & 0x0F).to(torch.int8)
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=q.device,
    )
    vals = lut[codes.to(torch.int64)]

    amax = torch.clamp(outer_amax.reshape(-1)[:rows], min=1e-12)
    inv_s = ((448.0 * 6.0) / amax).reciprocal().unsqueeze(1)
    return vals * (sf * inv_s).repeat_interleave(BLOCK_K, dim=1)


def both_directions(t):
    """Return (rowwise, columnwise^T) dequantized, both in (M, K) orientation."""
    dq_row = dequant(t._rowwise_data, t._rowwise_scale_inv, t._amax_rowwise)
    dq_col = dequant(t._columnwise_data, t._columnwise_scale_inv, t._amax_columnwise)
    return dq_row, dq_col.t()


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 3
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        print(f"FAIL: NVFP4 per-token needs SM100+, got SM{major}x")
        return 3

    failures = []
    for M, K in SHAPES:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        w = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

        # Leg A: per-token 1D (per-row outer amax).
        os.environ.pop(W_PER_TENSOR_1D_ENV, None)
        leg_a = quantize(make_quantizer(True, False, False), w)

        # Leg B: per-tensor 2D (the direction-independent control).
        leg_b = quantize(make_quantizer(True, True, False), w)
        row_b, col_b = both_directions(leg_b)

        # Leg C: per-tensor 1D (the new flag).
        os.environ[W_PER_TENSOR_1D_ENV] = "1"
        leg_c = quantize(make_quantizer(True, True, False), w)
        row_c, col_c = both_directions(leg_c)
        os.environ.pop(W_PER_TENSOR_1D_ENV, None)

        # Reference: plain per-tensor 1D weight (no per-token layout).
        per_tensor_1d = quantize(make_quantizer(False, False, False), w)

        tag = f"({M}x{K})"

        # 1. Leg A really is per-row (non-constant outer amax).
        amax_a = leg_a._amax_rowwise.reshape(-1)
        if bool(torch.all(amax_a == amax_a[0])):
            failures.append(f"{tag} leg A per-row outer amax is constant (expected per-row vector)")

        # 2. Leg B is direction-independent.
        if not torch.equal(row_b, col_b):
            frac = (row_b != col_b).float().mean().item()
            failures.append(f"{tag} leg B rowwise != columnwise^T on {frac:.2%} of elements")

        # 3. THE discriminating check: leg C is direction-DEPENDENT.
        frac_c = (row_c != col_c).float().mean().item()
        if frac_c <= 0.01:
            failures.append(
                f"{tag} leg C rowwise agrees with columnwise^T on {1 - frac_c:.2%} of "
                f"elements -- {W_PER_TENSOR_1D_ENV} looks like a no-op"
            )

        # 4. Leg C keeps the scalar outer amax broadcast (else it is just leg A).
        amax_c_row = leg_c._amax_rowwise.reshape(-1)
        amax_c_col = leg_c._amax_columnwise.reshape(-1)
        if not bool(torch.all(amax_c_row == amax_c_row[0])):
            failures.append(f"{tag} leg C rowwise outer amax is not constant")
        if not bool(torch.all(amax_c_col == amax_c_col[0])):
            failures.append(f"{tag} leg C columnwise outer amax is not constant")

        # 5. Leg C cast bytes == plain per-tensor 1D cast.
        if not torch.equal(
            per_tensor_1d._rowwise_data.view(torch.uint8), leg_c._rowwise_data.view(torch.uint8)
        ):
            failures.append(f"{tag} leg C rowwise FP4 data != per-tensor 1D reference")
        if not torch.equal(
            per_tensor_1d._rowwise_scale_inv.view(torch.uint8),
            leg_c._rowwise_scale_inv.view(torch.uint8),
        ):
            failures.append(f"{tag} leg C rowwise inner scale factors != per-tensor 1D reference")

        # 6. Legs B and C differ (independent no-op guard).
        if torch.equal(leg_b._rowwise_data.view(torch.uint8), leg_c._rowwise_data.view(torch.uint8)):
            failures.append(f"{tag} legs B and C produced identical FP4 data (flag is a no-op?)")

        print(
            f"{tag} legA_amax_row0={amax_a[0].item():.5f} "
            f"legB_dirdiff={(row_b != col_b).float().mean().item():.2%} "
            f"legC_dirdiff={frac_c:.2%} "
            f"legC_amax={amax_c_row[0].item():.5f}"
        )

    if failures:
        print("\n=== FAILED CHECKS ===")
        for f in failures:
            print("  -", f)
        return 3
    print("\n=== all three weight legs verified ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
