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
"""NVFP4 per-token W4A4 rollout support (no ModelOpt dependency).

This module is importable from BOTH sides of the refit boundary:

- Megatron training workers (mcore venv, **no vLLM installed**) use the
  weight producer and the refit iterator filter. Everything at module scope
  is therefore vLLM-free; vLLM imports live inside functions.
- vLLM generation workers use the registered ``nvfp4_pertoken`` quantization
  config (weights pre-quantized by the producer, activation global scales
  derived per token inside the FlashInfer TRT-LLM fused-MoE kernel).

The producer matches vLLM's online-quant kernel
(``vllm._custom_ops.scaled_fp4_quant`` as used by
``_quantize_moe_weight_to_nvfp4``) bit for bit; the unit test
``tests/unit/models/generation/test_nvfp4_pertoken_producer.py`` is the gate.
"""

import fnmatch
import logging
from collections.abc import Iterator
from typing import Any, Optional

import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Layers kept in native precision during rollout (mirrors the ModelOpt path's
# default; that path re-exports this constant so the dependency points from
# nemo_rl.modelopt -> here, never the reverse).
DEFAULT_NVFP4_IGNORE: list[str] = [
    "*lm_head*",
    # NOTE: the MoE router module prefix is exactly "...mlp.gate" — a trailing
    # ".*" would NOT fnmatch it and vLLM would NVFP4-quantize the router while
    # the refit streams it in BF16 (shape-mismatch at first refit).
    "*mlp.gate",
    "*mlp.gate.*",
    "*mlp.shared_expert*",
    "*self_attn*",
    "*embed_tokens*",
    "*input_layernorm*",
    "*post_attention_layernorm*",
    "*norm*",
]

class NvFp4PerTokenRolloutConfig(BaseModel, extra="allow"):
    """User config for the per-token NVFP4 W4A4 rollout.

    ``policy.generation.nvfp4_pertoken_rollout`` in YAML. Mutually exclusive
    with the ModelOpt QAT rollout keys (``quant_cfg`` / ``real_quant``).

    - ``enabled``: turn the mode on (quantized refit + per-token vLLM kernel).
    - ``ignore``: HF-name patterns kept in native precision during rollout;
      ``None`` uses :data:`DEFAULT_NVFP4_IGNORE`. Everything NOT ignored and
      matching ``quant_patterns`` is quantized at refit.
    - ``quant_patterns``: HF-name allowlist quantized at refit (MoE experts
      only — the per-token kernel is MoE-only).
    """

    enabled: bool = False
    ignore: Optional[list[str]] = None
    quant_patterns: list[str] = ["*.experts.*"]

    def resolved_ignore(self) -> list[str]:
        return list(DEFAULT_NVFP4_IGNORE) if self.ignore is None else self.ignore


_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_AMAX_DENOMINATOR = _FP4_MAX * _FP8_E4M3_MAX

# E2M1 representable magnitudes and rounding boundaries. Round-to-nearest-even
# on the grid: a value exactly on a boundary rounds toward the grid point with
# an even mantissa bit (0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0,
# 3.5->4.0, 5.0->4.0).
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Boundary indices whose tie resolves DOWN (toward the lower grid point).
_E2M1_TIE_DOWN = (0, 2, 4, 6)


def _round_e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    """Round |y| to E2M1 codes 0..7 with round-to-nearest-even semantics."""
    bounds = torch.tensor(_E2M1_BOUNDS, device=y.device, dtype=torch.float32)
    mag = y.abs()
    # searchsorted(right=True): ties land on the upper grid point...
    codes = torch.searchsorted(bounds, mag.reshape(-1).contiguous(), right=True)
    codes = codes.reshape(mag.shape).to(torch.uint8)
    # ...then push tie-down boundaries back to the lower point.
    for b_idx in _E2M1_TIE_DOWN:
        codes = torch.where(
            mag == _E2M1_BOUNDS[b_idx],
            torch.tensor(b_idx, device=y.device, dtype=torch.uint8),
            codes,
        )
    sign = (y < 0).to(torch.uint8) << 3
    # satfinite: values beyond the last boundary already clamp to code 7 (6.0)
    return sign | codes


def _quantize_blocks(scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-16 NVFP4 quantization of a pre-globally-scaled tensor.

    Mirrors ``scaled_fp4_quant(scaled, global_scale=1, non-swizzled)``:
    per-16-block e4m3 scale = RNE(block_amax / 6); elements are multiplied by
    the reciprocal of the decoded scale (multiply, not divide — matches the
    kernel) and rounded RNE onto the E2M1 grid, then nibble-packed with the
    even element in the low nibble.
    """
    *lead, k = scaled.shape
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"
    x = scaled.float().reshape(*lead, k // 16, 16)

    block_amax = x.abs().amax(dim=-1)
    block_scale = (block_amax / _FP4_MAX).to(torch.float8_e4m3fn)
    sf = block_scale.float()
    inv_sf = torch.where(sf > 0, sf.reciprocal(), torch.zeros_like(sf))
    y = x * inv_sf.unsqueeze(-1)

    codes = _round_e2m1_codes(y).reshape(*lead, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), block_scale.contiguous()


def quantize_nvfp4_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight to the NVFP4 (ModelOpt HF checkpoint) layout.

    Accepts ``(N, K)`` (one linear / one expert projection) or ``(E, N, K)``
    (stacked experts). Returns:

    - packed FP4 weight, uint8, ``(..., K // 2)``
    - block scales, float8_e4m3fn, ``(..., K // 16)``
    - global ``weight_scale_2``, float32, scalar for 2D / ``(E,)`` for 3D,
      stored as ``amax / (6 * 448)``

    Matches vLLM's ``_quantize_moe_weight_to_nvfp4`` numerics: per-tensor
    (per-expert) amax, global scale folded in with an intermediate cast back
    to the input dtype, then block-16 quantization under a unit global scale.
    """
    if weight.dim() == 2:
        amax = weight.abs().amax().float().clamp_min(1e-8)
        global_scale = _AMAX_DENOMINATOR / amax
        weight_scale_2 = (1.0 / global_scale).reshape(())
        scaled = (weight.float() * global_scale).to(weight.dtype)
    elif weight.dim() == 3:
        amax = weight.abs().amax(dim=(1, 2)).float().clamp_min(1e-8)
        global_scale = _AMAX_DENOMINATOR / amax
        weight_scale_2 = 1.0 / global_scale
        scaled = (weight.float() * global_scale[:, None, None]).to(weight.dtype)
    else:
        raise ValueError(f"expected 2D or 3D weight, got shape {tuple(weight.shape)}")

    packed, block_scale = _quantize_blocks(scaled)
    return packed, block_scale, weight_scale_2.to(torch.float32)


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def iter_nvfp4_pertoken_weights(
    base_iter: Iterator[tuple[str, torch.Tensor]],
    quant_patterns: list[str],
    ignore_patterns: Optional[list[str]] = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Refit filter: quantize matching ``*.weight`` tensors in an export stream.

    Wraps the ``(hf_name, tensor)`` iterator the Megatron policy worker already
    produces (TP-gathered, HF-named). Matching weights are quantized on their
    current device (on-GPU before IPC -> 4x smaller transfers) and yielded in
    the ModelOpt NVFP4 HF checkpoint layout::

        <base>.weight, <base>.weight_scale, <base>.weight_scale_2

    Everything else (non-matching weights, biases, kv scales, draft weights)
    passes through untouched.
    """
    ignore = ignore_patterns or []
    quantized = 0
    passthrough = 0
    for name, tensor in base_iter:
        if (
            not name.endswith(".weight")
            or not _matches_any(name, quant_patterns)
            or _matches_any(name, ignore)
        ):
            passthrough += 1
            yield name, tensor
            continue
        packed, block_scale, weight_scale_2 = quantize_nvfp4_weight(tensor)
        base = name[: -len(".weight")]
        quantized += 1
        yield name, packed
        yield f"{base}.weight_scale", block_scale
        yield f"{base}.weight_scale_2", weight_scale_2

    # Per-refit liveness proof: a config/name mismatch (e.g. quant_patterns
    # not matching the export's expert naming) would otherwise silently
    # degrade to an all-BF16 refit that vLLM then fails to load — or worse.
    logger.info(
        "[nvfp4_pertoken] refit: quantized %d params -> %d tensors, passthrough %d",
        quantized,
        3 * quantized,
        passthrough,
    )
    if quant_patterns and quantized == 0:
        raise RuntimeError(
            "[nvfp4_pertoken] refit quantized 0 params although quant_patterns="
            f"{quant_patterns} is configured — export naming and patterns are "
            "out of sync."
        )


def build_nvfp4_pertoken_hf_quant_config(ignore: list[str]) -> dict[str, Any]:
    """HF ``quantization_config`` override for the per-token W4A4 rollout.

    A literal dict (no ModelOpt conversion helper): NVFP4 weights with
    block-16 e4m3 scales; activations dynamic (per-token global scales are
    derived inside the kernel, no ``input_scale`` tensors exist).
    """
    # Mirrors the quantization_config of ModelOpt NVFP4 HF checkpoints
    # (e.g. nvidia/Qwen3-30B-A3B-NVFP4 config.json) key-for-key — vLLM's
    # ModelOpt config parser is shape-sensitive (`ignore`, not
    # `exclude_modules`; `targets` inside the group). Only delta:
    # input_activations.dynamic=True since no input_scale tensors exist.
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt"},
        "ignore": list(ignore),
        "config_groups": {
            "group_0": {
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
    }
