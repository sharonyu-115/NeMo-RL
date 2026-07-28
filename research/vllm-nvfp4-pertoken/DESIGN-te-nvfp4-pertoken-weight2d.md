# Design: NVFP4 per-token fwd + bwd with **2D weights** (`NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1`) in the RL flow

Branch: `shuangy/te-nvfp4-pertoken-backward` (worktree `nemo-rl-te-bwd`), TE fork
`cael-ling@690ffea4` (PR #3045). Extends
[`DESIGN-te-nvfp4-pertoken-backward.md`]. Weight-quantization theory and the
reachability matrix: `session/te-nvfp4-weight-quant-legs/handoff.md`.

## TL;DR

**The plumbing already exists and needs no new config surface.** `Fp4Config.per_token_weight_2d`
is already translated to `NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1`
(`nemo_rl/models/megatron/fp4_env.py:_FP4_PER_TOKEN_FLAG_ENV`) and injected into the Megatron
worker's runtime env on the driver (`lm_policy.py:151`). **No TE rebuild is required** — unlike the
per-tensor-1D leg, the weight-2D cast ships in the already-baked `nemo-rl-te690ffea-probe.sqsh`.

What is missing is **three correctness guards, one recipe, one launcher mode, and validation**.
Sections G1–G3 below are real defects that bite specifically when the weight-2D flag is used;
they are the substance of this design.

## 1. What the flag actually does (verified in TE @690ffea)

| layer | file:line | behavior |
|---|---|---|
| recipe | `common/recipe/__init__.py:583` | `per_token_weight_2d = os.getenv("NVTE_NVFP4_PER_TOKEN_WEIGHT_2D","0")=="1"` on the **base** `NVFP4BlockScaling` — so Megatron's default recipe construction picks it up with no code change |
| quantizer select | `pytorch/quantization.py:1734-1737` | applies to the **forward `weight` slot only**, and only when `per_token` is already true |
| ctor guards | `tensor/nvfp4_tensor.py:212-223` | requires `per_token`; rejects RHT/SR **on the weight quantizer** (both are hard-off there anyway — `fp4_quant_fwd_weight` pins `random_hadamard_transform=False, stochastic_rounding=False`) |
| cast | `csrc/quantizer.cpp:2536-2606` | scalar amax → per-tensor 2D cast (`nvte_quantize_v2`, 16×16 inner) → broadcast the scalar across the `(M,)`/`(K,)` per-token amax vectors so the per-token CUTLASS GEMM consumes it unchanged. bf16-only, no noop_flag, no amax reduction |
| repr | `common/recipe/__init__.py:685-686` | prints `per_token_weight_2d=True` in the recipe log line |

Numerics: 2D inner tiles + a scalar outer amax make the quantized weight **transposition
invariant**, so forward (rowwise) and dgrad (columnwise) consume the *same* weight. The per-token
1D default does not: `W_row ≠ W_colᵀ`, a fixed function of `W` that is a **bias**, not noise. That
bias is only *paid* when the backward is real FP4, which is why this flag is interesting mainly in
combination with `backward: nvfp4_pertoken`. Cost: one scale per 256 elements instead of per 16.

### 1.1 It is safe on the MoE-expert path — but only by default

`csrc/extensions/cast.cpp:1611-1615` **hard-rejects** `per_token_weight_2d` in `split_quantize`
("grouped 2D weight cast is not implemented"). This does *not* fire in our recipe because
`GroupedLinear` quantizes expert weights **one at a time** via `quantize_weight`
(`module/grouped_linear.py:208-222`); `split_quantize` only ever receives *activation* and
*grad_output* quantizers (`:539`, `:958-965`, `:1079`). The fused grouped-tensor path that would
route weights through the grouped kernel is gated behind
`NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM` (default `0`, `grouped_linear.py:117-118`) and
additionally requires `with_rht` on the input quantizers (`:139-140`), which per-token defaults off.
→ **Safe today, one env var away from a hard crash.** See G3.

## 2. Gaps in the RL flow

### G1 — `per_token_*` flags silently switch a forward-only leg into FP4 backward *(defect)*

`fp4_env.py:fp4_cfg_wants_per_token_backward` returns true when **any** `per_token_*` flag is set,
independent of `backward`. `megatron_policy_worker.py:410-415` consumes that predicate to force
`self._nvte_backward_override = None` and `os.environ.pop("NVTE_BACKWARD_OVERRIDE")`.

So adding `per_token_weight_2d: true` to the **forward-only** leg (`-fp4fwd`, which inherits
`NVTE_BACKWARD_OVERRIDE: dequantized`) silently converts it into an FP4-backward run — the exact
variable the leg is supposed to hold fixed. The driver-side warning does not fire either, because
`to_unset` stays empty for `backward != "nvfp4_pertoken"`.

**Fix** (`fp4_env.py`): split the predicate.

```python
def fp4_cfg_wants_per_token_backward(fp4_cfg) -> bool:
    """True only when the config asks for real FP4 per-token dgrad/wgrad."""
    return bool(fp4_cfg) and fp4_cfg.get("enabled", False) and \
        fp4_cfg.get("backward") == "nvfp4_pertoken"


def fp4_cfg_uses_per_token(fp4_cfg) -> bool:
    """True when any per-token feature is requested (capability-gate scope)."""
    return fp4_cfg_wants_per_token_backward(fp4_cfg) or (
        bool(fp4_cfg) and fp4_cfg.get("enabled", False)
        and any(fp4_cfg.get(f) for f in _FP4_PER_TOKEN_FLAG_ENV)
    )
```

`assert_te_supports_fp4_backward` keeps the broad predicate (`fp4_cfg_uses_per_token`); the worker's
override-clearing at `megatron_policy_worker.py:396` switches to the narrow one. This makes
`weight_2d` orthogonal to `backward`, which is required for the fwd-only ablation leg in §4.

### G2 — `per_token_weight_2d` is a silent no-op without per-token forward *(defect)*

The flag is inert unless `NVTE_NVFP4_PER_TOKEN=1` (`quantization.py:1735` requires `per_token`).
TE emits no warning; the run looks configured and trains as the plain per-token-1D (or even
row-scaled) leg. Given that the campaign's parent recipe pins `NVTE_NVFP4_PER_TOKEN: "1"` in raw
`env_vars` while the `-fp4bwd` leg gets it from the typed `backward` field, both spellings are live
and it is easy to land on neither.

**Fix** (`fp4_env.py:apply_fp4_backward_env_overrides`, after the merge): if any
`NVTE_NVFP4_PER_TOKEN_*` var ended up set but the *effective* `env_vars.get("NVTE_NVFP4_PER_TOKEN") != "1"`,
raise `ValueError` naming both spellings. Checking post-merge is what makes it correct — it sees the
raw-`env_vars` spelling and the typed-field spelling identically.

### G3 — weight-2D + fused grouped GEMM is an unguarded hard crash *(low severity, cheap guard)*

If `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1` is ever set (perf experiment, container default
change), expert weights route through `split_quantize` → `NVTE_CHECK` abort at `cast.cpp:1614`,
mid-run, from C++. **Fix**: in the same post-merge validation, reject
`per_token_weight_2d` together with `env_vars.get("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM") == "1"`
with a message pointing at `cast.cpp:1614`.

### G4 — observability

`apply_fp4_backward_env_overrides` already prints a `[fp4_cfg]` wiring line; extend it with an
explicit weight-cast label so runs are greppable and not confused with the per-tensor-1D leg that
the sibling session proposes (`session/te-nvfp4-weight-quant-legs/handoff.md`), which will *also*
print `per_token_weight_2d=True` in TE's repr:

```
weight_cast=per-tensor-2D (scalar outer amax, 16x16 inner)
```

## 3. Interaction matrix (verified, not assumed)

| combination | status |
|---|---|
| `backward: nvfp4_pertoken` + `per_token_weight_2d` | **the target leg.** Supported; removes the fwd/dgrad weight mismatch |
| per-token fwd only (`NVTE_BACKWARD_OVERRIDE=dequantized`) + `weight_2d` | supported *after G1*; isolates the pure forward-resolution effect |
| `per_token_sr` + `weight_2d` | **compatible.** SR lands on `fp4_quant_bwd_grad`, never the weight; the weight quantizer's SR is hard-off so the ctor guard cannot trip |
| `per_token_rht` + `weight_2d` | do **not** run. Not a TE conflict (weight RHT is hard-off) but RHT is broken on this build — swizzle-layout `validate_encode_output` assert, HEAD `4c21e2062`; and RHT would additionally arm the fused-grouped path in G3 |
| `weight_2d` + grouped-expert checkpoint save | expected fine — `install_te_grouped_empty_extra_state_patch()` (`megatron_policy_worker.py:391`) is keyed on `fp4_cfg.enabled`, not on the weight mode. **Verify in the smoke** (save at step 1) |
| `weight_2d` + vLLM refit | no interaction. Master weights stay BF16; refit re-quantizes independently |

## 4. The train/gen angle — why this is more than a gradient-bias ablation

The rollout quantizes weights with a **per-tensor (per-expert) global scale + 1D 16-element block
scales** (`nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py:147-178`:
`amax.abs().amax()` → `weight_scale_2`, then `_quantize_blocks` over the last dim).

| leg | trainer weight cast | outer scale vs vLLM | inner geometry vs vLLM |
|---|---|---|---|
| per-token default (1D) | per-row outer, 16-elem inner | ✗ (vector vs scalar) | ✓ |
| **`WEIGHT_2D=1`** | **scalar outer, 16×16 inner** | **✓** | ✗ |
| per-tensor 1D (needs the sibling-session TE patch + bake) | scalar outer, 16-elem inner | ✓ | ✓ |

So `WEIGHT_2D=1` trades one axis of train/gen weight mismatch for another. It is therefore a
**two-hypothesis experiment**: (a) does removing the fwd/dgrad transposition bias improve
convergence, and (b) does matching the rollout's outer-scale *shape* move `token_mult_prob_error` /
`gen_kl`. Judge both; do not attribute a `gen_kl` change to the gradient bias alone.

Caveat on (b): with `expert_model_parallel_size: 8` and ETP defaulting to `tensor_model_parallel_size: 2`,
TE's scalar amax is taken over the **local TP shard** of each expert weight, while vLLM's is over the
**full** expert weight (refit gathers before quantizing). Alignment is approximate, not exact.

## 5. Changes

| File | Change |
|---|---|
| `nemo_rl/models/megatron/fp4_env.py` | G1 predicate split; G2+G3 post-merge validation; G4 log label |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` | use the narrow predicate at `:396` for override-clearing; keep the broad one for the capability gate at `:386` |
| `nemo_rl/models/policy/__init__.py` | docstring only — `per_token_weight_2d` now explicitly orthogonal to `backward`, and requires per-token forward |
| `tests/unit/models/megatron/test_fp4_env.py` | weight-2D-only config emits `WEIGHT_2D=1` and **no** `NVTE_BACKWARD_OVERRIDE` unset; raises without per-token; raises with fused-grouped-GEMM |
| `examples/configs/recipes/llm/...-nvfp4-pertoken-fp4bwd-r3-w2d.yaml` (new) | leg: inherits `-fp4bwd-r3`, adds `per_token_weight_2d: true` |
| `research/vllm-nvfp4-pertoken/run_dapo_longrun.sh` | `PRECISION=nvfp4_bwd_w2d` (te690 image guard, same as `nvfp4_bwd`) |

No TE change, no rebuild, no Megatron change.

### Recipe (leg on top of the running fp4bwd-r3 baseline)

```yaml
defaults: grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3.yaml
# fp4bwd-r3 + 2D weight cast. Typed field (not a raw env var) so the driver logs the
# wiring line and the G2 per-token validation runs; RHT/SR stay OFF.
checkpointing:
  checkpoint_dir: results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3-w2d
policy:
  megatron_cfg:
    fp4_cfg:
      per_token_weight_2d: true
logger:
  wandb:
    name: grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3-w2d
```

Typed field over raw env var here (the `-rhtsr` legs used raw vars): the typed path is what carries
the G2/G3 validation and the wiring log, and weight-2D is the flag most likely to be silently inert.

## 6. Validation

1. **Unit** — `uv run pytest tests/unit/models/megatron/test_fp4_env.py`.
2. **Wiring assertion, no GPU** — construct `NVFP4BlockScaling()` under
   `NVTE_NVFP4_PER_TOKEN=1 NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1` in the probe image and assert the repr
   shows `per_token=True, per_token_weight_2d=True, backward_override=None`.
3. **Discriminating numeric check** (cheap, single GPU, borrow the harness in
   `research/vllm-nvfp4-pertoken/diag_rht_sr_forward_probe.py`): quantize one weight with and
   without the flag, dequantize rowwise and columnwise. **With the flag, rowwise must equal
   columnwiseᵀ; without it they must differ.** If they are equal in both cases the flag never
   engaged (G2) — this is the assertion that catches a silent no-op.
4. **1-step smoke** — `PRECISION=nvfp4_bwd_w2d MAX_STEPS=1` with a checkpoint save, to clear the
   grouped-expert `_extra_state` path with the 2D weight cast.
5. **Leg run** — 1500 steps under `chain_keeper.sh`, A/B against the in-flight `-fp4bwd-r3` run
   (identical data/topology/seed; only the weight cast differs). Metrics:
   `token_mult_prob_error`, `gen_kl`, `grad_norm`, loss — per §4, read `gen_kl` and the gradient
   metrics as separate hypotheses.

## 7. Risks

- **Untested territory.** `WEIGHT_2D=1` has never been run here, with or without FP4 backward
  (P3 smoke was default per-token weights). Budget the smoke.
- **Not a guaranteed win.** Coarser scales (1 per 256 vs 1 per 16) may cost more than the removed
  bias buys, especially in the forward. A regression is a valid, publishable result for the leg.
- **Naming collision with the sibling session.** Both this leg and the proposed per-tensor-1D leg
  print `per_token_weight_2d=True` in TE's repr. G4's log label is what disambiguates them; do not
  skip it if both legs will run.
- `690ffea4` is an unmerged fork HEAD that may force-push.
