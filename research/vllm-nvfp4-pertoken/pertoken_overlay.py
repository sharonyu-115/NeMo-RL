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
"""Prototype: per-token NVFP4 activation scales on a pre-quantized checkpoint.

Registers quantization method ``modelopt_fp4_pertoken``: identical to vLLM's
stock ``modelopt_fp4`` W4A4 path except FusedMoE layers ignore the
checkpoint's calibrated ``input_scale`` and let the FlashInfer TRT-LLM kernel
derive per-token activation global scales at runtime (the vllm#48538
``Nvfp4OnlineMoEMethod`` machinery, applied to externally-quantized weights).
This is the vLLM analog of SGLang's ``SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE=1``
and the exact activation regime NeMo-RL's per-token w4a4 wiring will use on
top of PR #2983 (Megatron-exported quantized weights, no input scales).

Classes live at module scope: the resolved quantization config is pickled to
vLLM's EngineCore process, so it must be importable there (this module must be
on PYTHONPATH of the launching process; child procs inherit it).

Usage:
    from pertoken_overlay import register_modelopt_fp4_pertoken
    register_modelopt_fp4_pertoken()
    llm = LLM(model=<modelopt-nvfp4-ckpt>, quantization="modelopt_fp4_pertoken")

Dense/attention layers keep the stock static-input-scale ModelOpt path; the
per-token change is MoE-experts-only, mirroring the NeMo-RL quant scope.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

PER_TOKEN_METHOD = "modelopt_fp4_pertoken"

_registered = False


class ModelOptNvFp4PerTokenFusedMoE(ModelOptNvFp4FusedMoE):
    """W4A4 MoE with checkpoint input scales replaced by per-token quant."""

    def __init__(self, quant_config, moe_config) -> None:
        super().__init__(quant_config, moe_config)
        if self.use_a16:
            raise ValueError(
                "modelopt_fp4_pertoken requires a W4A4 NVFP4 checkpoint, "
                "got W4A16_NVFP4."
            )
        # make_nvfp4_moe_kernel silently drops per_token_activation for
        # every backend except FLASHINFER_TRTLLM — fail loudly instead of
        # running with stale static scales.
        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
            raise ValueError(
                "modelopt_fp4_pertoken requires the FlashInfer TRT-LLM MoE "
                f"backend, got {self.nvfp4_backend}."
            )

    def process_weights_after_loading(self, layer) -> None:
        # Discard the checkpoint's calibrated activation scales. Neutral
        # (1.0) global scales make the kernel's output scalars reduce to
        # the weight scales; per-token scales are derived at runtime
        # (same trick as Nvfp4OnlineMoEMethod._quantize_weights).
        num_experts = layer.w13_input_scale.data.shape[0]
        device = layer.w13_weight.device
        ones = torch.ones(num_experts, device=device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())
        logger.info_once(
            "modelopt_fp4_pertoken: ignoring checkpoint input_scale; "
            "per-token NVFP4 activation scaling active"
        )

        # Below mirrors ModelOptNvFp4FusedMoE.process_weights_after_loading
        # except make_nvfp4_moe_kernel(per_token_activation=True).
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()

        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        # FlashInfer backends may return activation global scales as stride-0
        # expanded views. vLLM's layerwise reload finalize does
        # param.data.copy_() into every kernel tensor, which raises on
        # stride-0 storage ("more than one element ... refers to a single
        # memory location") — the same hazard PR #2983 handles with
        # .contiguous() in its registered method. Contiguous is a no-op for
        # already-dense tensors.
        def _dense(t):
            return t.contiguous() if isinstance(t, torch.Tensor) else t

        replace_parameter(layer, "w13_weight", _dense(w13))
        replace_parameter(layer, "w13_weight_scale", _dense(w13_scale))
        replace_parameter(layer, "w13_weight_scale_2", _dense(w13_scale_2))
        replace_parameter(layer, "w13_input_scale", _dense(a13_scale))
        replace_parameter(layer, "w2_weight", _dense(w2))
        replace_parameter(layer, "w2_weight_scale", _dense(w2_scale))
        replace_parameter(layer, "w2_weight_scale_2", _dense(w2_scale_2))
        replace_parameter(layer, "w2_input_scale", _dense(a2_scale))

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
            per_token_activation=True,
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)


class ModelOptNvFp4PerTokenConfig(ModelOptNvFp4Config):
    """Stock ModelOpt NVFP4 config with per-token FusedMoE activations."""

    FusedMoEMethodCls = ModelOptNvFp4PerTokenFusedMoE

    def get_name(self):
        return PER_TOKEN_METHOD

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Never auto-select from checkpoint metadata; only explicit
        # --quantization modelopt_fp4_pertoken picks this config.
        if user_quant == PER_TOKEN_METHOD:
            return PER_TOKEN_METHOD
        return None


def register_modelopt_fp4_pertoken() -> None:
    """Register the per-token ModelOpt NVFP4 config through vLLM's public API."""
    global _registered
    if _registered:
        return
    register_quantization_config(PER_TOKEN_METHOD)(ModelOptNvFp4PerTokenConfig)
    _registered = True
    logger.info("Registered vLLM quantization method %r", PER_TOKEN_METHOD)
