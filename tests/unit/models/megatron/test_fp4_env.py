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
"""Unit tests for the typed Fp4Config -> NVTE_* per-token-backward translation.

The module under test is dependency-light (stdlib only), so it imports directly
without the megatron/TE extras.
"""

import importlib.util

import pytest

from nemo_rl.models.megatron.fp4_env import (
    apply_fp4_backward_env_overrides,
    assert_te_supports_fp4_backward,
    fp4_cfg_to_env_overrides,
    fp4_cfg_wants_per_token_backward,
)


# ------------------------------------------------ fp4_cfg_to_env_overrides


def test_absent_backward_emits_nothing():
    # An fp4 recipe without the typed backward field must translate to nothing,
    # so it keeps TE's own backward default (byte-for-byte preservation).
    to_set, to_unset = fp4_cfg_to_env_overrides(
        {"enabled": True, "fp4": "e2m1", "fp4_recipe": "nvfp4"}
    )
    assert to_set == {}
    assert to_unset == set()


def test_disabled_cfg_emits_nothing():
    assert fp4_cfg_to_env_overrides({"enabled": False, "backward": "nvfp4_pertoken"}) == (
        {},
        set(),
    )
    assert fp4_cfg_to_env_overrides(None) == ({}, set())


def test_backward_dequantized():
    to_set, to_unset = fp4_cfg_to_env_overrides(
        {"enabled": True, "backward": "dequantized"}
    )
    assert to_set == {"NVTE_BACKWARD_OVERRIDE": "dequantized"}
    assert to_unset == set()


def test_backward_high_precision():
    to_set, to_unset = fp4_cfg_to_env_overrides(
        {"enabled": True, "backward": "high_precision"}
    )
    assert to_set == {"NVTE_BACKWARD_OVERRIDE": "high_precision"}
    assert to_unset == set()


def test_backward_nvfp4_pertoken_sets_switch_and_unsets_override():
    to_set, to_unset = fp4_cfg_to_env_overrides(
        {"enabled": True, "backward": "nvfp4_pertoken"}
    )
    assert to_set == {"NVTE_NVFP4_PER_TOKEN": "1"}
    assert to_unset == {"NVTE_BACKWARD_OVERRIDE"}


def test_per_token_flags():
    to_set, _ = fp4_cfg_to_env_overrides(
        {
            "enabled": True,
            "backward": "nvfp4_pertoken",
            "per_token_rht": True,
            "per_token_sr": False,
            "per_token_weight_2d": True,
        }
    )
    assert to_set["NVTE_NVFP4_PER_TOKEN"] == "1"
    assert to_set["NVTE_NVFP4_PER_TOKEN_RHT"] == "1"
    assert to_set["NVTE_NVFP4_PER_TOKEN_WEIGHT_2D"] == "1"
    assert "NVTE_NVFP4_PER_TOKEN_SR" not in to_set  # False -> not emitted


def test_invalid_backward_raises():
    with pytest.raises(ValueError, match="Invalid fp4_cfg.backward"):
        fp4_cfg_to_env_overrides({"enabled": True, "backward": "fp4_magic"})


# ------------------------------------------ apply_fp4_backward_env_overrides


def test_apply_injects_into_env_vars():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    apply_fp4_backward_env_overrides(env, {"enabled": True, "backward": "dequantized"})
    assert env["NVTE_BACKWARD_OVERRIDE"] == "dequantized"
    # Existing vars untouched.
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_apply_raw_env_vars_win():
    # Recipe pins NVTE_BACKWARD_OVERRIDE=high_precision; typed backward asks for
    # dequantized. Raw wins (D3).
    env = {"NVTE_BACKWARD_OVERRIDE": "high_precision"}
    apply_fp4_backward_env_overrides(env, {"enabled": True, "backward": "dequantized"})
    assert env["NVTE_BACKWARD_OVERRIDE"] == "high_precision"


def test_apply_pertoken_absent_override_left_unset():
    env = {}
    apply_fp4_backward_env_overrides(
        env, {"enabled": True, "backward": "nvfp4_pertoken"}
    )
    assert env["NVTE_NVFP4_PER_TOKEN"] == "1"
    assert "NVTE_BACKWARD_OVERRIDE" not in env


def test_apply_pertoken_clears_inherited_override():
    # An inherited/pinned NVTE_BACKWARD_OVERRIDE is CLEARED (typed backward is
    # authoritative for the unset) with a warning, so per-token backward engages.
    env = {"NVTE_BACKWARD_OVERRIDE": "dequantized"}
    with pytest.warns(UserWarning, match="requires NVTE_BACKWARD_OVERRIDE UNSET"):
        apply_fp4_backward_env_overrides(
            env, {"enabled": True, "backward": "nvfp4_pertoken"}
        )
    assert "NVTE_BACKWARD_OVERRIDE" not in env
    assert env["NVTE_NVFP4_PER_TOKEN"] == "1"


# --------------------------------------------- capability helpers / gate


def test_wants_per_token_backward():
    assert fp4_cfg_wants_per_token_backward(
        {"enabled": True, "backward": "nvfp4_pertoken"}
    )
    assert fp4_cfg_wants_per_token_backward(
        {"enabled": True, "per_token_sr": True}
    )
    assert not fp4_cfg_wants_per_token_backward(
        {"enabled": True, "backward": "dequantized"}
    )
    assert not fp4_cfg_wants_per_token_backward({"enabled": False})
    assert not fp4_cfg_wants_per_token_backward(None)


def test_gate_noop_when_not_requested():
    # Must not import TE nor raise when per-token backward is not requested.
    assert assert_te_supports_fp4_backward({"enabled": True, "backward": "dequantized"}) is None
    assert assert_te_supports_fp4_backward(None) is None


@pytest.mark.skipif(
    importlib.util.find_spec("transformer_engine") is not None,
    reason="TransformerEngine is installed; capability gate would not raise",
)
def test_gate_raises_when_te_absent():
    with pytest.raises(RuntimeError, match="PR #3045"):
        assert_te_supports_fp4_backward(
            {"enabled": True, "backward": "nvfp4_pertoken"}
        )
