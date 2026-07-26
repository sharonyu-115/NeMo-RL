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
"""Monkeypatch for Megatron grouped-expert checkpoint save under NVFP4 per-token.

`NVTE_NVFP4_PER_TOKEN=1` (used by the NVFP4 per-token BACKWARD recipe, TE PR #3045)
makes the grouped MoE-expert linear's ``_extra_state`` come back EMPTY — per-token
FP4 computes scales on-the-fly, so there is legitimately no persistent scale/amax
state to serialize (unlike delayed scaling). Megatron's
``TEGroupedLinear._split_extra_state`` does not guard that case: it sees the fp8
flag set, decodes the empty ``_extra_state`` to ``None``, then subscripts it:

    megatron/core/extensions/transformer_engine.py:_split_extra_state
      state = self._decode_extra_state(state)   # empty tensor -> None
      extra_fp8_variables = state["extra_fp8_variables"]   # None[...] -> TypeError

The row-scaled forward path populates ``_extra_state`` and saves fine, so this only
bites per-token mode. The LOAD side (``merge_extra_states``) already guards
``_decode_extra_state(...) is None`` and early-returns, so this patch just makes the
SAVE side symmetric: replicate the (empty) ``_extra_state`` per gemm, exactly like
the existing ``not fp8_checkpoint`` fallback. It is a no-op for any module whose
``_extra_state`` is non-empty (normal/delayed fp8), so it is safe to install
unconditionally on the Megatron worker.
"""

from functools import wraps

_PATCH_ATTR = "_nrl_empty_extra_state_patch"


def install_te_grouped_empty_extra_state_patch() -> None:
    """Idempotently patch TEGroupedLinear._split_extra_state to tolerate empty _extra_state.

    Safe no-op if TE/grouped linear is unavailable in this build.
    """
    try:
        from megatron.core.extensions.transformer_engine import TEGroupedLinear
    except ImportError:
        return  # TE grouped linear not built into this Megatron — nothing to patch.

    original_split = TEGroupedLinear._split_extra_state
    if getattr(original_split, _PATCH_ATTR, False):
        return  # already installed

    @wraps(original_split)
    def _split_extra_state_empty_safe(self, state):
        # Mirror the upstream guard expression.
        fp8_checkpoint = (
            self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration
        )
        # The ONLY divergence from upstream: when the module is fp8/fp4-flagged but
        # its serialized _extra_state is empty (decodes to None), fall back to the
        # same shape the no-fp8 branch returns instead of subscripting None. The
        # load-side merge_extra_states already tolerates this per-gemm empty state.
        if fp8_checkpoint and self._decode_extra_state(state) is None:
            return [state] * self.num_gemms
        return original_split(self, state)

    setattr(_split_extra_state_empty_safe, _PATCH_ATTR, True)
    TEGroupedLinear._split_extra_state = _split_extra_state_empty_safe
