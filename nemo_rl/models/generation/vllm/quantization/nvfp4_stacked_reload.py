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
"""Experimental layer-stacked reload for NeMo's per-token NVFP4 rollout.

This module is imported only when ``experimental_stacked_reload=true`` selects
``NvFp4PerTokenStackedWorkerExtension``.  The production worker extension and
its per-expert reload stream are left untouched.

The experiment deliberately keeps quantization identical: it wraps the proven
``NvFp4PerTokenQuantizer`` and coalesces its already-quantized per-expert output
into the eight checkpoint-shaped tensors owned by a ``RoutedExperts`` layer.
Only the matching layer instances and their recorded reload metadata receive a
stacked loader.  No installed vLLM source or process-global class is patched.
"""

import re
import time
from collections.abc import Iterable, Iterator
from itertools import chain
from types import MethodType
from typing import Optional
from weakref import WeakSet

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken import (
    ModelOptNvFp4PerTokenFusedMoE,
    NvFp4PerTokenQuantizer,
    NvFp4PerTokenWorkerExtension,
)
from nemo_rl.models.generation.vllm.vllm_backend import _IPCReloadMetrics

_QUANTIZED_EXPERT_PARAM_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\."
    r"(?P<eid>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_scale_2|weight_scale|input_scale|weight)$"
)

_KINDS = ("weight", "weight_scale", "weight_scale_2", "input_scale")
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

_STACKED_PARAM_BY_NAME = {
    "gate_up_proj.weight": "w13_weight",
    "gate_up_proj.weight_scale": "w13_weight_scale",
    "gate_up_proj.weight_scale_2": "w13_weight_scale_2",
    "gate_up_proj.input_scale": "w13_input_scale",
    "down_proj.weight": "w2_weight",
    "down_proj.weight_scale": "w2_weight_scale",
    "down_proj.weight_scale_2": "w2_weight_scale_2",
    "down_proj.input_scale": "w2_input_scale",
}

_LEGACY_LOAD_WEIGHTS = RoutedExperts.load_weights
_LEGACY_WEIGHT_LOADER = RoutedExperts.weight_loader
_PATCHED_LAYERS: WeakSet[RoutedExperts] = WeakSet()


def _normalize_stacked_name(name: str) -> str:
    """Normalize names passed either by FusedMoE or RoutedExperts directly."""
    return name.removeprefix("experts.")


def _stacked_weight_loader(
    self: RoutedExperts,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
    stacked_param_name: str | None = None,
) -> bool | None:
    """Load one full pre-kernel parameter, or delegate to vLLM unchanged."""
    if stacked_param_name is None:
        return _LEGACY_WEIGHT_LOADER(
            self,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=return_success,
        )

    expected_param = getattr(self, stacked_param_name)
    if param is not expected_param:
        raise RuntimeError(
            "NVFP4 stacked reload routed "
            f"{weight_name!r} to {stacked_param_name!r}, but the supplied "
            "parameter is a different object"
        )
    if tuple(param.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            "NVFP4 stacked reload shape mismatch for "
            f"{self.layer_name}.{stacked_param_name}: expected "
            f"{tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
        )

    # This copy_ is intentional. vLLM's layerwise reload first invokes this
    # loader under CopyCounter on meta tensors, then replays it on materialized
    # tensors. One full-parameter copy replaces all per-expert loader calls.
    param.data.copy_(loaded_weight)
    return True if return_success else None


def _stacked_load_weights(
    self: RoutedExperts, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterator[str]:
    """Recognize an all-stacked stream, otherwise use vLLM's original loader."""
    iterator = iter(weights)
    try:
        first = next(iterator)
    except StopIteration:
        return

    first_name = _normalize_stacked_name(first[0])
    if first_name not in _STACKED_PARAM_BY_NAME:
        # Dummy/ordinary checkpoint loading, or a deliberately ignored BF16
        # expert layer. Preserve the original one-pass implementation.
        yield from _LEGACY_LOAD_WEIGHTS(self, chain((first,), iterator))
        return

    for expert_name, loaded_weight in chain((first,), iterator):
        normalized_name = _normalize_stacked_name(expert_name)
        param_name = _STACKED_PARAM_BY_NAME.get(normalized_name)
        if param_name is None:
            raise RuntimeError(
                "NVFP4 stacked reload received a mixed stacked/per-expert "
                f"stream for {self.layer_name}: unexpected {expert_name!r}"
            )

        param = getattr(self, param_name)
        success = param.weight_loader(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=normalized_name,
            shard_id="stacked",
            expert_id=-1,
            return_success=True,
            stacked_param_name=param_name,
        )
        if success:
            yield param_name


def _require_supported_topology(layer: RoutedExperts) -> None:
    parallel = layer.moe_config.moe_parallel_config
    if parallel.tp_size != 1 or parallel.ep_size != 1 or parallel.enable_eplb:
        raise ValueError(
            "experimental NVFP4 stacked reload currently requires "
            "TP=1, EP=1, and EPLB disabled; got "
            f"TP={parallel.tp_size}, EP={parallel.ep_size}, "
            f"EPLB={parallel.enable_eplb} for {layer.layer_name}"
        )


def install_stacked_reload_on_model(model: torch.nn.Module) -> dict[str, int]:
    """Patch only per-token NVFP4 RoutedExperts instances and reload metadata."""
    expected_experts: dict[str, int] = {}
    for module in model.modules():
        if not isinstance(module, RoutedExperts) or not isinstance(
            module.quant_method, ModelOptNvFp4PerTokenFusedMoE
        ):
            continue

        _require_supported_topology(module)
        expected_experts[module.layer_name] = module.global_num_experts
        if module in _PATCHED_LAYERS:
            continue

        info = get_layerwise_info(module)
        restore_params, _ = info.restore_metadata
        missing = sorted(set(_STACKED_PARAM_BY_NAME.values()) - restore_params.keys())
        if missing:
            raise RuntimeError(
                f"NVFP4 stacked reload metadata for {module.layer_name} is "
                f"missing checkpoint-shaped parameters: {missing}"
            )

        module.load_weights = MethodType(_stacked_load_weights, module)
        for param_name in set(_STACKED_PARAM_BY_NAME.values()):
            restore_params[param_name].weight_loader = MethodType(
                _stacked_weight_loader, module
            )
        _PATCHED_LAYERS.add(module)

    if not expected_experts:
        raise RuntimeError(
            "experimental NVFP4 stacked reload found no per-token NVFP4 "
            "RoutedExperts layers"
        )
    return expected_experts


class NvFp4LayerCoalescer:
    """Combine verified per-expert NVFP4 outputs into layer-stacked tensors."""

    def __init__(self, expected_experts: dict[str, int]) -> None:
        self._expected_experts = dict(expected_experts)
        self._pending: dict[str, dict[tuple[int, str, str], torch.Tensor]] = {}
        self._stacked_layers = 0
        self._coalesce_seconds = 0.0
        self.reset()

    def reset(self) -> None:
        self._pending = {}
        self._stacked_layers = 0
        self._coalesce_seconds = 0.0

    @property
    def stacked_layers(self) -> int:
        return self._stacked_layers

    @property
    def coalesce_seconds(self) -> float:
        return self._coalesce_seconds

    def process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        start = time.perf_counter()
        out = self._process(weights)
        self._coalesce_seconds += time.perf_counter() - start
        return out

    def _process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        out: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            match = _QUANTIZED_EXPERT_PARAM_RE.match(name)
            if match is None:
                out.append((name, tensor))
                continue

            layer_prefix = match.group("prefix")
            num_experts = self._expected_experts.get(layer_prefix)
            if num_experts is None:
                # An additional_ignore layer remains BF16 and uses vLLM's
                # original per-expert loader.
                out.append((name, tensor))
                continue

            key = (
                int(match.group("eid")),
                match.group("proj"),
                match.group("kind"),
            )
            bucket = self._pending.setdefault(layer_prefix, {})
            if key in bucket:
                raise RuntimeError(
                    f"NVFP4 stacked reload received duplicate tensor {name!r}"
                )
            bucket[key] = tensor

            expected_parts = num_experts * len(_PROJECTIONS) * len(_KINDS)
            if len(bucket) == expected_parts:
                out.extend(self._flush_layer(layer_prefix, num_experts, bucket))
                del self._pending[layer_prefix]
                self._stacked_layers += 1
            elif len(bucket) > expected_parts:
                raise RuntimeError(
                    f"NVFP4 stacked reload collected too many tensors for "
                    f"{layer_prefix}: {len(bucket)} > {expected_parts}"
                )
        return out

    @staticmethod
    def _flush_layer(
        layer_prefix: str,
        num_experts: int,
        bucket: dict[tuple[int, str, str], torch.Tensor],
    ) -> list[tuple[str, torch.Tensor]]:
        def part(expert_id: int, projection: str, kind: str) -> torch.Tensor:
            key = (expert_id, projection, kind)
            try:
                return bucket[key]
            except KeyError as error:
                raise RuntimeError(
                    "NVFP4 stacked reload cannot flush "
                    f"{layer_prefix}; missing expert {expert_id} "
                    f"{projection}.{kind}"
                ) from error

        out: list[tuple[str, torch.Tensor]] = []
        for kind in _KINDS:
            if kind in ("weight", "weight_scale"):
                w13 = torch.stack(
                    [
                        torch.cat(
                            (
                                part(eid, "gate_proj", kind),
                                part(eid, "up_proj", kind),
                            ),
                            dim=0,
                        )
                        for eid in range(num_experts)
                    ],
                    dim=0,
                )
            else:
                w13 = torch.stack(
                    [
                        torch.stack(
                            (
                                part(eid, "gate_proj", kind).reshape(()),
                                part(eid, "up_proj", kind).reshape(()),
                            )
                        )
                        for eid in range(num_experts)
                    ],
                    dim=0,
                )

            w2 = torch.stack(
                [part(eid, "down_proj", kind) for eid in range(num_experts)],
                dim=0,
            )
            out.append((f"{layer_prefix}.gate_up_proj.{kind}", w13))
            out.append((f"{layer_prefix}.down_proj.{kind}", w2))
        return out

    def finish(self) -> None:
        if self._pending:
            details = {
                prefix: len(parts) for prefix, parts in sorted(self._pending.items())
            }
            raise RuntimeError(
                f"NVFP4 stacked reload ended with incomplete layers: {details}"
            )
        if self._stacked_layers == 0:
            raise RuntimeError("NVFP4 stacked reload emitted 0 stacked layers")


class StackedNvFp4ReloadPreparer:
    """Keep current quantization and change only the emitted reload granularity."""

    def __init__(
        self,
        quantizer: NvFp4PerTokenQuantizer,
        expected_experts: dict[str, int],
    ) -> None:
        self.quantizer = quantizer
        self.coalescer = NvFp4LayerCoalescer(expected_experts)

    def reset(self) -> None:
        self.quantizer.reset()
        self.coalescer.reset()

    def process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        return self.coalescer.process(self.quantizer.process(weights))

    def finish(self) -> None:
        self.quantizer.finish()
        self.coalescer.finish()


class NvFp4PerTokenStackedWorkerExtension(NvFp4PerTokenWorkerExtension):
    """Default-off POC using layer-stacked checkpoint-format NVFP4 tensors."""

    _stacked_preparer: Optional[StackedNvFp4ReloadPreparer] = None

    def _get_reload_weight_preparer(self) -> StackedNvFp4ReloadPreparer:
        if self._stacked_preparer is None:
            expected_experts = install_stacked_reload_on_model(self.model_runner.model)
            self._stacked_preparer = StackedNvFp4ReloadPreparer(
                self._get_quantizer(), expected_experts
            )
        return self._stacked_preparer

    def _on_ipc_reload_complete(self, metrics: _IPCReloadMetrics) -> None:
        super()._on_ipc_reload_complete(metrics)
        preparer = self._get_reload_weight_preparer()
        print(
            "[nvfp4_pertoken][stacked-experimental] refit: stacked "
            f"{preparer.coalescer.stacked_layers} layers | coalesce_enqueue "
            f"{preparer.coalescer.coalesce_seconds:.2f}s",
            flush=True,
        )
