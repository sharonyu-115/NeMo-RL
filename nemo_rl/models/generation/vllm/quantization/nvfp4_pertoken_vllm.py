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
"""vLLM-side pieces of the NVFP4 per-token W4A4 rollout.

Split from ``nvfp4_pertoken.py`` deliberately: the classes here subclass vLLM
types and must live at module scope (the resolved quantization config is
pickled to vLLM's EngineCore process), which requires module-scope vLLM
imports — so this module is only importable inside the vLLM environment.
The producer/refit-filter side stays vLLM-free in ``nvfp4_pertoken.py``.

Validated by the standalone probe (research/vllm-nvfp4-pertoken/): identical
to vLLM's stock ``modelopt_fp4`` W4A4 path except FusedMoE layers ignore any
checkpoint ``input_scale`` and build the per-token dynamic kernel
(``make_nvfp4_moe_kernel(per_token_activation=True)``); kernel scale tensors
are made contiguous so layerwise-reload finalize can ``copy_()`` into them.
"""

from contextlib import contextmanager
from typing import Any, Iterator

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

from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken import (
    build_nvfp4_pertoken_hf_quant_config,
)
from nemo_rl.models.generation.vllm.vllm_backend import (
    VllmInternalWorkerExtension,
    WeightUpdateFinalizer,
    WeightUpdateTransport,
)

logger = init_logger(__name__)

NVFP4_PER_TOKEN_METHOD = "nvfp4_pertoken"

_registered = False


class ModelOptNvFp4PerTokenFusedMoE(ModelOptNvFp4FusedMoE):
    """W4A4 MoE: pre-quantized weights, per-token dynamic activation scales.

    The class NAME must contain "ModelOpt": vLLM's RoutedExperts.weight_loader
    duck-types NVFP4 scale loading on ``"ModelOpt" in
    self.quant_method.__class__.__name__`` (routed_experts.py); a rename
    silently drops expert scale params out of that branch and initial load
    fails with "quant method must be one of ['tensor','channel','group',
    'block']".
    """

    def __init__(self, quant_config, moe_config) -> None:
        super().__init__(quant_config, moe_config)
        if self.use_a16:
            raise ValueError(
                f"{NVFP4_PER_TOKEN_METHOD} requires a W4A4 NVFP4 checkpoint, "
                "got W4A16_NVFP4."
            )
        # make_nvfp4_moe_kernel silently drops per_token_activation for every
        # backend except FLASHINFER_TRTLLM — fail loudly instead of running
        # with stale static scales.
        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
            raise ValueError(
                f"{NVFP4_PER_TOKEN_METHOD} requires the FlashInfer TRT-LLM MoE "
                f"backend, got {self.nvfp4_backend}."
            )

    def process_weights_after_loading(self, layer) -> None:
        # Neutral (1.0) global activation scales: the kernel derives per-token
        # scales at runtime, so the output scalars reduce to the weight scales.
        num_experts = layer.w13_input_scale.data.shape[0]
        device = layer.w13_weight.device
        ones = torch.ones(num_experts, device=device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())
        logger.info_once(
            "%s: per-token NVFP4 activation scaling active", NVFP4_PER_TOKEN_METHOD
        )

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

        # Stride-0 expanded scale views break layerwise-reload finalize
        # (param.data.copy_() into broadcast storage); contiguous is a no-op
        # for already-dense tensors. Probe finding, matches PR #2983.
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


class NvFp4PerTokenConfig(ModelOptNvFp4Config):
    """Stock ModelOpt NVFP4 config with per-token FusedMoE activations."""

    FusedMoEMethodCls = ModelOptNvFp4PerTokenFusedMoE

    def get_name(self):
        return NVFP4_PER_TOKEN_METHOD

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Never auto-select from checkpoint metadata; only an explicit
        # quantization="nvfp4_pertoken" picks this config.
        if user_quant == NVFP4_PER_TOKEN_METHOD:
            return NVFP4_PER_TOKEN_METHOD
        return None


def register_nvfp4_pertoken() -> None:
    """Register the per-token NVFP4 config through vLLM's public API."""
    global _registered
    if _registered:
        return
    register_quantization_config(NVFP4_PER_TOKEN_METHOD)(NvFp4PerTokenConfig)
    _registered = True
    logger.info("Registered vLLM quantization method %r", NVFP4_PER_TOKEN_METHOD)


class NvFp4PerTokenWorkerExtension(VllmInternalWorkerExtension):
    """Refit transport for per-token NVFP4 rollouts.

    The refit stream carries pre-quantized per-expert tensors in the ModelOpt
    NVFP4 HF checkpoint layout (produced by ``iter_nvfp4_pertoken_weights``),
    which vLLM's NVFP4 loaders consume natively — no fused-family batching.
    Weight updates run inside vLLM's layerwise reload lifecycle so quantized
    params are restored to load format before loading and re-processed
    (per-token kernel rebuilt) afterwards, preserving CUDA-graph-stable
    kernel storage.
    """

    @contextmanager
    def _weight_update_lifecycle(
        self, transport: WeightUpdateTransport
    ) -> Iterator[WeightUpdateFinalizer]:
        del transport
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self.model_runner.model

        def finalize() -> None:
            with torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)
            torch.accelerator.synchronize()

        with torch.device(self.device):
            initialize_layerwise_reload(model)
        yield finalize

    def _weight_update_errors_are_fatal(self) -> bool:
        return True

    def _synchronize_before_ipc_data_ack(self) -> None:
        torch.accelerator.synchronize()


def configure_nvfp4_pertoken_engine_kwargs(
    llm_kwargs: dict[str, Any], ignore: list[str]
) -> None:
    """Mutate vLLM engine kwargs for the per-token W4A4 rollout.

    - registers and selects the ``nvfp4_pertoken`` quantization method
    - overrides the HF quantization config (weights NVFP4, activations dynamic)
    - dummy initial load: params are NVFP4-shaped and the BF16 checkpoint on
      disk cannot fill them; the first refit (which always precedes the first
      generation) provides every weight
    - installs the refit worker extension
    """
    register_nvfp4_pertoken()
    llm_kwargs["quantization"] = NVFP4_PER_TOKEN_METHOD
    llm_kwargs["load_format"] = "dummy"
    hf_overrides = llm_kwargs.setdefault("hf_overrides", {})
    hf_overrides["quantization_config"] = build_nvfp4_pertoken_hf_quant_config(ignore)
    llm_kwargs["worker_extension_cls"] = (
        "nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_vllm."
        "NvFp4PerTokenWorkerExtension"
    )
