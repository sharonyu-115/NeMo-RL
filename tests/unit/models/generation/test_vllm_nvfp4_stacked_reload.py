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
"""Focused tests for the default-off NVFP4 stacked-reload experiment."""

import types
import weakref

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    """This CPU-only module does not need the suite-wide Ray cluster."""
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():
    """This CPU-only module does not collect suite-wide GPU metrics."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data):
    """Avoid querying Ray GPU metadata during this CPU-only module."""
    yield _unit_test_data


@pytest.fixture()
def stacked_module():
    pytest.importorskip("vllm")
    from nemo_rl.models.generation.vllm.quantization import (
        nvfp4_stacked_reload as M,
    )

    return M


def _per_expert_outputs(prefix: str, num_experts: int):
    pieces = []
    by_name = {}
    for expert_id in range(num_experts):
        for projection_index, projection in enumerate(
            ("gate_proj", "up_proj", "down_proj")
        ):
            value = 100 * expert_id + 10 * projection_index
            if projection == "down_proj":
                weight = torch.full((4, 2), value + 1, dtype=torch.float32)
                scale = torch.full((4, 1), value + 2, dtype=torch.float32)
            else:
                weight = torch.full((2, 4), value + 1, dtype=torch.float32)
                scale = torch.full((2, 1), value + 2, dtype=torch.float32)
            values = {
                "weight": weight,
                "weight_scale": scale,
                "weight_scale_2": torch.tensor(float(value + 3)),
                "input_scale": torch.tensor(float(value + 4)),
            }
            for kind, tensor in values.items():
                name = f"{prefix}.{expert_id}.{projection}.{kind}"
                pieces.append((name, tensor))
                by_name[name] = tensor
    return pieces, by_name


def test_coalescer_losslessly_stacks_one_layer(stacked_module):
    M = stacked_module
    prefix = "model.layers.2.mlp.experts"
    pieces, source = _per_expert_outputs(prefix, num_experts=2)
    coalescer = M.NvFp4LayerCoalescer({prefix: 2})

    # One complete expert is insufficient; the coalescer owns those tensors
    # until the entire layer can be emitted.
    assert coalescer.process(pieces[:12]) == []
    stacked = dict(coalescer.process(pieces[12:]))
    coalescer.finish()

    assert set(stacked) == {
        f"{prefix}.{projection}.{kind}"
        for projection in ("gate_up_proj", "down_proj")
        for kind in M._KINDS
    }
    for kind in M._KINDS:
        expected_w13 = []
        expected_w2 = []
        for expert_id in range(2):
            gate = source[f"{prefix}.{expert_id}.gate_proj.{kind}"]
            up = source[f"{prefix}.{expert_id}.up_proj.{kind}"]
            down = source[f"{prefix}.{expert_id}.down_proj.{kind}"]
            if kind in ("weight", "weight_scale"):
                expected_w13.append(torch.cat((gate, up), dim=0))
            else:
                expected_w13.append(torch.stack((gate.reshape(()), up.reshape(()))))
            expected_w2.append(down)

        assert torch.equal(
            stacked[f"{prefix}.gate_up_proj.{kind}"],
            torch.stack(expected_w13),
        )
        assert torch.equal(
            stacked[f"{prefix}.down_proj.{kind}"],
            torch.stack(expected_w2),
        )

    assert coalescer.stacked_layers == 1
    assert coalescer.coalesce_seconds > 0.0


def test_coalescer_preserves_ignored_layer_and_rejects_partial_layer(
    stacked_module,
):
    M = stacked_module
    quantized_prefix = "model.layers.2.mlp.experts"
    ignored_name = "model.layers.0.mlp.experts.0.down_proj.weight"
    ignored_weight = torch.randn(4, 4)
    coalescer = M.NvFp4LayerCoalescer({quantized_prefix: 2})

    passthrough = coalescer.process([(ignored_name, ignored_weight)])
    assert passthrough == [(ignored_name, ignored_weight)]

    pieces, _ = _per_expert_outputs(quantized_prefix, num_experts=2)
    assert coalescer.process(pieces[:1]) == []
    with pytest.raises(RuntimeError, match="incomplete layers"):
        coalescer.finish()


def test_stacked_loader_uses_one_full_parameter_copy(stacked_module):
    M = stacked_module
    param = torch.nn.Parameter(torch.empty(2, 3), requires_grad=False)
    layer = types.SimpleNamespace(w13_weight=param)
    calls = []

    def weight_loader(**kwargs):
        calls.append(kwargs)
        return M._stacked_weight_loader(layer, **kwargs)

    param.weight_loader = weight_loader
    loaded_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    loaded_names = list(
        M._stacked_load_weights(
            layer,
            [("gate_up_proj.weight", loaded_weight)],
        )
    )

    assert loaded_names == ["w13_weight"]
    assert len(calls) == 1
    assert calls[0]["stacked_param_name"] == "w13_weight"
    assert torch.equal(param, loaded_weight)


def test_nonstacked_stream_delegates_to_legacy_loader(stacked_module, monkeypatch):
    M = stacked_module
    received = []

    def legacy_loader(_layer, weights):
        received.extend(weights)
        yield "legacy"

    monkeypatch.setattr(M, "_LEGACY_LOAD_WEIGHTS", legacy_loader)
    weights = [
        ("0.gate_proj.weight", torch.randn(2, 4)),
        ("0.up_proj.weight", torch.randn(2, 4)),
    ]

    assert list(M._stacked_load_weights(object(), weights)) == ["legacy"]
    assert received == weights


def test_stacked_loader_rejects_mixed_stream(stacked_module):
    M = stacked_module
    param = torch.nn.Parameter(torch.empty(2, 3), requires_grad=False)
    layer = types.SimpleNamespace(w13_weight=param, layer_name="layer")
    param.weight_loader = lambda **_kwargs: True

    with pytest.raises(RuntimeError, match="mixed stacked/per-expert"):
        list(
            M._stacked_load_weights(
                layer,
                [
                    ("gate_up_proj.weight", torch.empty(2, 3)),
                    ("0.down_proj.weight", torch.empty(2, 3)),
                ],
            )
        )


def test_installer_patches_only_matching_layer_instance(stacked_module, monkeypatch):
    M = stacked_module

    class FakeRoutedExperts:
        def load_weights(self, weights):
            yield from weights

        def weight_loader(self, **_kwargs):
            return None

    class FakePerTokenQuantMethod:
        pass

    parallel = types.SimpleNamespace(tp_size=1, ep_size=1, enable_eplb=False)
    layer = FakeRoutedExperts()
    layer.quant_method = FakePerTokenQuantMethod()
    layer.moe_config = types.SimpleNamespace(moe_parallel_config=parallel)
    layer.layer_name = "model.layers.2.mlp.experts"
    layer.global_num_experts = 2
    unrelated_layer = FakeRoutedExperts()
    unrelated_layer.quant_method = object()

    original_class_loader = FakeRoutedExperts.load_weights
    restore_params = {
        name: torch.nn.Parameter(torch.empty(1), requires_grad=False)
        for name in set(M._STACKED_PARAM_BY_NAME.values())
    }
    info = types.SimpleNamespace(restore_metadata=(restore_params, {}))
    model = types.SimpleNamespace(modules=lambda: [layer, unrelated_layer])

    monkeypatch.setattr(M, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(M, "ModelOptNvFp4PerTokenFusedMoE", FakePerTokenQuantMethod)
    monkeypatch.setattr(M, "get_layerwise_info", lambda _layer: info)
    monkeypatch.setattr(M, "_PATCHED_LAYERS", weakref.WeakSet())

    assert M.install_stacked_reload_on_model(model) == {layer.layer_name: 2}
    assert FakeRoutedExperts.load_weights is original_class_loader
    assert "load_weights" in layer.__dict__
    assert "load_weights" not in unrelated_layer.__dict__
    assert all(
        param.weight_loader.__self__ is layer for param in restore_params.values()
    )
