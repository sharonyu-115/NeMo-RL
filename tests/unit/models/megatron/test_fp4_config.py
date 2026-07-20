"""Unit tests for the TE NVFP4 training config port (fp4_cfg).

Self-contained: loads the helpers directly so the tests run without the
megatron/vLLM extras (mirrors test_nvfp4_pertoken_producer.py's approach).
"""

import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest


def _load_apply_te_precision_config():
    try:
        from nemo_rl.models.megatron.setup import apply_te_precision_config

        return apply_te_precision_config
    except ImportError:
        # Extract just the helper function without importing the module's
        # heavy dependency graph.
        path = (
            pathlib.Path(__file__).resolve().parents[4]
            / "nemo_rl/models/megatron/setup.py"
        )
        import ast

        tree = ast.parse(path.read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "apply_te_precision_config"
        )
        mod = ast.Module(body=[fn], type_ignores=[])
        ns: dict = {"PolicyConfig": dict}
        exec(compile(mod, str(path), "exec"), ns)  # noqa: S102 - test-only
        return ns["apply_te_precision_config"]


apply_te = _load_apply_te_precision_config()


def _model_cfg():
    return SimpleNamespace(
        fp8=None,
        fp8_recipe=None,
        fp8_param=None,
        fp4=None,
        fp4_recipe=None,
        fp4_param=None,
    )


def _config(megatron_cfg):
    return {"megatron_cfg": megatron_cfg}


def test_fp4_cfg_sets_transformer_config_fields():
    mc = _model_cfg()
    apply_te(
        mc,
        _config(
            {
                "fp4_cfg": {
                    "enabled": True,
                    "fp4": "e2m1",
                    "fp4_recipe": "nvfp4",
                    "fp4_param": False,
                }
            }
        ),
    )
    assert mc.fp4 == "e2m1"
    assert mc.fp4_recipe == "nvfp4"
    assert mc.fp4_param is False
    assert mc.fp8 is None


def test_fp8_and_fp4_both_enabled_raises():
    with pytest.raises(ValueError, match="cannot both"):
        apply_te(
            _model_cfg(),
            _config(
                {
                    "fp8_cfg": {"enabled": True},
                    "fp4_cfg": {"enabled": True},
                }
            ),
        )


def test_fp4_missing_key_raises_keyerror():
    with pytest.raises(KeyError, match="fp4_cfg"):
        apply_te(
            _model_cfg(),
            _config({"fp4_cfg": {"enabled": True, "fp4": "e2m1"}}),
        )


def test_fp4_custom_recipe_requires_factory():
    with pytest.raises(ValueError, match="fp4_quantizer_factory"):
        apply_te(
            _model_cfg(),
            _config(
                {
                    "fp4_cfg": {
                        "enabled": True,
                        "fp4": "e2m1",
                        "fp4_recipe": "custom",
                        "fp4_param": False,
                    }
                }
            ),
        )


def test_fp4_disabled_leaves_model_cfg_untouched():
    mc = _model_cfg()
    apply_te(mc, _config({"fp4_cfg": {"enabled": False}}))
    assert mc.fp4 is None


def test_f2l4_passthrough():
    mc = _model_cfg()
    apply_te(
        mc,
        _config(
            {
                "first_last_layers_bf16": True,
                "num_layers_at_start_in_bf16": 2,
                "num_layers_at_end_in_bf16": 4,
            }
        ),
    )
    assert mc.first_last_layers_bf16 is True
    assert mc.num_layers_at_start_in_bf16 == 2
    assert mc.num_layers_at_end_in_bf16 == 4


def test_te_precision_config_file_loads_recipe(monkeypatch):
    sentinel = object()
    fake_utils = types.ModuleType("megatron.core.quantization.utils")
    fake_utils.load_quantization_recipe = lambda path: sentinel
    fake_quant = types.ModuleType("megatron.core.quantization")
    fake_core = types.ModuleType("megatron.core")
    fake_megatron = types.ModuleType("megatron")
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(sys.modules, "megatron.core.quantization", fake_quant)
    monkeypatch.setitem(sys.modules, "megatron.core.quantization.utils", fake_utils)

    mc = _model_cfg()
    apply_te(mc, _config({"te_precision_config_file": "/tmp/recipe.yaml"}))
    assert mc.quant_recipe is sentinel


# ---------------------------------------------- NVTE_BACKWARD_OVERRIDE gating


def _load_worker_ctx():
    """Instantiate just the override-gating logic from the worker via __new__."""
    try:
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorker,
        )

        return MegatronPolicyWorker
    except Exception:
        pytest.skip("full nemo_rl deps unavailable")


def test_nvte_backward_override_train_only(monkeypatch):
    cls = _load_worker_ctx()
    worker = cls.__new__(cls)
    monkeypatch.setenv("NVTE_BACKWARD_OVERRIDE", "dequantized")
    # replicate the __init__ capture logic
    worker._nvte_backward_override = os.environ.get("NVTE_BACKWARD_OVERRIDE")
    os.environ.pop("NVTE_BACKWARD_OVERRIDE", None)

    assert "NVTE_BACKWARD_OVERRIDE" not in os.environ
    with worker._nvte_backward_override_training_ctx():
        assert os.environ["NVTE_BACKWARD_OVERRIDE"] == "dequantized"
    assert "NVTE_BACKWARD_OVERRIDE" not in os.environ

    worker._nvte_backward_override = None
    with worker._nvte_backward_override_training_ctx():
        assert "NVTE_BACKWARD_OVERRIDE" not in os.environ
