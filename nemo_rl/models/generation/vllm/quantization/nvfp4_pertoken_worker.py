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
"""Generation workers for the per-token NVFP4 W4A4 rollout.

Thin subclasses of the standard vLLM workers that configure the engine for
the ``nvfp4_pertoken`` quantization method at creation time. Selected via
``resolve_generation_worker_cls`` when
``generation.nvfp4_pertoken_rollout.enabled`` is set.
"""

from typing import Any

import ray

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken import (
    NvFp4PerTokenRolloutConfig,
)
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


def _configure_pertoken_engine_kwargs(
    cfg: VllmConfig, llm_kwargs: dict[str, Any]
) -> None:
    rollout_cfg = NvFp4PerTokenRolloutConfig.model_validate(
        cfg.get("nvfp4_pertoken_rollout") or {}
    )
    assert rollout_cfg.enabled, (
        "per-token worker selected but nvfp4_pertoken_rollout.enabled is False"
    )
    # vLLM-side module: only importable inside the vLLM environment.
    from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_vllm import (
        configure_nvfp4_pertoken_engine_kwargs,
    )

    configure_nvfp4_pertoken_engine_kwargs(llm_kwargs, rollout_cfg.resolved_ignore())


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)  # pragma: no cover
class NvFp4PerTokenGenerationWorker(VllmGenerationWorkerImpl):
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        _configure_pertoken_engine_kwargs(self.cfg, llm_kwargs)
        super()._create_engine(llm_kwargs)


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class NvFp4PerTokenAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        _configure_pertoken_engine_kwargs(self.cfg, llm_kwargs)
        super()._create_engine(llm_kwargs)
