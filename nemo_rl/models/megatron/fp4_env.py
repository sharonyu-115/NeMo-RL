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
"""Typed Fp4Config -> NVTE_* env-var translation for NVFP4 per-token backward.

Kept dependency-light (stdlib only; TransformerEngine imported lazily in the
capability gate) so it can be imported on the driver — where the Megatron
worker's runtime ``env_vars`` are assembled — as well as inside the worker.

Wiring (TransformerEngine PR #3045, per-token backward):
  * The derived NVTE_* vars are injected into the Megatron worker's runtime
    ``env_vars`` on the driver (nemo_rl.models.policy.lm_policy), so they land in
    the worker's os.environ at process start — the same channel today's forward
    knobs and NVTE_BACKWARD_OVERRIDE ride. This is early enough for the worker's
    train-only NVTE_BACKWARD_OVERRIDE capture (MegatronPolicyWorker.__init__).
  * The capability gate + the per-token override-disable run inside the worker,
    where TransformerEngine is importable.
"""

import warnings
from typing import Optional

# Typed Fp4Config per-token-backward flag fields -> NVTE_NVFP4_PER_TOKEN_* env
# vars. `backward` is handled separately because it also drives an UNSET.
_FP4_PER_TOKEN_FLAG_ENV = {
    "per_token_rht": "NVTE_NVFP4_PER_TOKEN_RHT",
    "per_token_sr": "NVTE_NVFP4_PER_TOKEN_SR",
    "per_token_weight_2d": "NVTE_NVFP4_PER_TOKEN_WEIGHT_2D",
}


def fp4_cfg_to_env_overrides(
    fp4_cfg: Optional[dict],
) -> tuple[dict[str, str], set[str]]:
    """Translate typed Fp4Config backward/per-token fields to NVTE_* env vars.

    Pure (no side effects) for unit-testability. Returns ``(to_set, to_unset)``.
    Only fields that are EXPLICITLY present produce output: an absent ``backward``
    emits nothing, so existing fp4 recipes that rely on TE's own backward default
    keep byte-for-byte behavior. The caller applies these with raw
    ``megatron_cfg.env_vars`` winning (design D3).

    Mapping (see nemo_rl.models.policy.Fp4Config):
      backward="dequantized"    -> set NVTE_BACKWARD_OVERRIDE=dequantized
      backward="high_precision" -> set NVTE_BACKWARD_OVERRIDE=high_precision
      backward="nvfp4_pertoken" -> set NVTE_NVFP4_PER_TOKEN=1, UNSET NVTE_BACKWARD_OVERRIDE
      per_token_{rht,sr,weight_2d}=True -> set NVTE_NVFP4_PER_TOKEN_{RHT,SR,WEIGHT_2D}=1
    """
    to_set: dict[str, str] = {}
    to_unset: set[str] = set()
    if not fp4_cfg or not fp4_cfg.get("enabled", False):
        return to_set, to_unset

    backward = fp4_cfg.get("backward")
    if backward == "dequantized":
        to_set["NVTE_BACKWARD_OVERRIDE"] = "dequantized"
    elif backward == "high_precision":
        to_set["NVTE_BACKWARD_OVERRIDE"] = "high_precision"
    elif backward == "nvfp4_pertoken":
        to_set["NVTE_NVFP4_PER_TOKEN"] = "1"
        to_unset.add("NVTE_BACKWARD_OVERRIDE")
    elif backward is not None:
        raise ValueError(
            f"Invalid fp4_cfg.backward={backward!r}; expected one of "
            "'dequantized', 'high_precision', 'nvfp4_pertoken'."
        )

    for field, env_name in _FP4_PER_TOKEN_FLAG_ENV.items():
        if fp4_cfg.get(field):
            to_set[env_name] = "1"
    return to_set, to_unset


def apply_fp4_backward_env_overrides(
    env_vars: dict[str, str], fp4_cfg: Optional[dict]
) -> dict[str, str]:
    """Merge NVFP4-backward NVTE_* overrides into a worker runtime ``env_vars``.

    For SET vars, raw ``env_vars`` (from the recipe) WIN over typed-field-derived
    values (design D3): a var the recipe pins is left as-is. For the per-token
    UNSET, the typed ``backward='nvfp4_pertoken'`` is authoritative — the derived
    UNSET removes any pinned/inherited NVTE_BACKWARD_OVERRIDE (else it would
    silently suppress the feature the field just enabled, and inheritance can't
    drop a nested key). Mutates and returns ``env_vars``.
    """
    to_set, to_unset = fp4_cfg_to_env_overrides(fp4_cfg)
    for k in to_unset:
        if k in env_vars:
            warnings.warn(
                f"[fp4_cfg] backward='nvfp4_pertoken' requires {k} UNSET; clearing "
                f"megatron_cfg.env_vars {k}={env_vars[k]!r} so FP4 per-token backward "
                "engages (the typed backward field is authoritative for this).",
                stacklevel=2,
            )
        env_vars.pop(k, None)
    for k, v in to_set.items():
        env_vars.setdefault(k, v)  # raw env_vars win for SET vars
    if to_set or to_unset:
        print(
            f"[fp4_cfg] NVFP4 backward wiring: backward={fp4_cfg.get('backward')!r} "
            f"-> set={to_set} unset={sorted(to_unset)} "
            f"(effective NVTE_NVFP4_PER_TOKEN={env_vars.get('NVTE_NVFP4_PER_TOKEN')!r}, "
            f"NVTE_BACKWARD_OVERRIDE={env_vars.get('NVTE_BACKWARD_OVERRIDE')!r})",
            flush=True,
        )
    return env_vars


def fp4_cfg_wants_per_token_backward(fp4_cfg: Optional[dict]) -> bool:
    """True if the config requests NVFP4 per-token backward (or a per-token flag)."""
    if not fp4_cfg or not fp4_cfg.get("enabled", False):
        return False
    return fp4_cfg.get("backward") == "nvfp4_pertoken" or any(
        fp4_cfg.get(f) for f in _FP4_PER_TOKEN_FLAG_ENV
    )


def assert_te_supports_fp4_backward(fp4_cfg: Optional[dict]) -> None:
    """Fail loudly if per-token backward is requested but TE lacks PR #3045.

    The per-token backward recipe (NVFP4PerTokenBlockScaling) only exists in the
    PR #3045 line; its importability is a clean capability signal that the
    installed TransformerEngine understands the NVTE_NVFP4_PER_TOKEN switch.
    Call inside the worker, where TransformerEngine is importable.
    """
    if not fp4_cfg_wants_per_token_backward(fp4_cfg):
        return
    try:
        from transformer_engine.common.recipe import (  # noqa: F401
            NVFP4PerTokenBlockScaling,
        )
    except ImportError as e:
        raise RuntimeError(
            "fp4_cfg requests NVFP4 per-token backward (backward='nvfp4_pertoken' "
            "or per_token_* flags), but the installed TransformerEngine lacks "
            "PR #3045 (NVFP4PerTokenBlockScaling not importable). Rebuild the "
            "worker venv on a TransformerEngine that carries the per-token "
            "backward recipe."
        ) from e
