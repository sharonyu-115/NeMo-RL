# Design: TE NVFP4 per-token W4A4 — train → refit → vLLM rollout (no ModelOpt)

Status: draft (2026-07-19). Companion to this directory's probe (`README.md`),
which validated the vLLM-side kernel path, fidelity win, and reload contract.

## Goal

A quantized-rollout flow where:

1. **Training** runs real NVFP4 compute via TE (Megatron `fp4_cfg`, per-token
   activation scaling) — no ModelOpt quantizers in the model, BF16 master weights.
2. **Refit** quantizes the BF16 master weights to NVFP4 (vLLM/FlashInfer tensor
   layout) at export time, every step — no calibration anywhere.
3. **Rollout** runs vLLM w4a4 with per-token dynamic activation global scales
   (the probe's validated path).

Hard requirements: **independent of the ModelOpt QAT path** (no imports from
`nemo_rl/modelopt/`, no TensorQuantizers, no quantized checkpoint caches), and
**maximum reuse** of existing machinery.

## Architectural principle: follow the fp8-rollout pattern, not the QAT pattern

NeMo-RL already has two quantized-rollout shapes:

| | ModelOpt QAT (#2983) | fp8 rollouts (main) |
|---|---|---|
| Worker | dedicated `MegatronQuantPolicyWorker` (model surgery: `mtq.quantize`, calib, quantizer state in ckpt) | plain `MegatronPolicyWorker` |
| Refit payload | quantized (packed w4 + scales), exported via bridge modelopt exporter | BF16 (+ kv scales), quantized vLLM-side at load |
| vLLM | registered ModelOpt configs + `VllmQuantInternalWorkerExtension` | base extension + `quantization/fp8.py` |

The QAT path needs a dedicated worker **because quantizers live inside the
training model**. TE-native FP4 training has no such state — quantization is in
TE's compute path, configured like `fp8_cfg`. So this design hangs off the
**base** `MegatronPolicyWorker` like fp8 does, with one difference inherited
from #2983: the refit payload is *pre-quantized* (4x smaller transfers, and
rollout weights are exactly what the producer kernel emits).

```
════════ STARTUP ════════
 Megatron (base MegatronPolicyWorker)        vLLM (thin worker extension)
┌─────────────────────────────────┐   ┌──────────────────────────────────────┐
│ fp4_cfg → TE NVFP4 recipe       │   │ quantization="nvfp4_pertoken"        │
│ (per-token activations, f2l4)   │   │ (registered config; graduated probe  │
│ NO quantize/calib/ckpt-cache    │   │  overlay) + literal hf quant dict    │
└─────────────────────────────────┘   │ create_weights: FP4 params, NO       │
                                      │ input scales                         │
════════ EVERY STEP ════════          └──────────────────────────────────────┘
 1. TRAIN  TE NVFP4 GEMMs (real quant compute); BF16 master weights updated
 2. REFIT  bridge.export_hf_weights_nvfp4(models, quantize_fn=<producer>)
             ├ .weight          packed FP4 (E,N,K/2) uint8
             ├ .weight_scale    block-16 E4M3
             └ .weight_scale_2  per-expert FP32 (amax/(448·6), from master wt)
           — no input scales, no draft changes, kv scales as today —
           → ZMQ IPC / collective, layerwise reload lifecycle (as #2983)
 3. ROLLOUT FlashInfer TRT-LLM fused MoE, per_token_activation=True
```

## Components and reuse map

### 1. Training — `policy.megatron_cfg.fp4_cfg` (mostly exists)

Upstream the rl-fp4 study's TE NVFP4 knobs (`fp4_cfg`, per-module recipe,
`f2l4`, optional 4over6) into nemo-rl main, mirroring how `fp8_cfg` is read
(`megatron_policy_worker.py:373`). New v2 config class:

```python
class Fp4Config(BaseModel, extra="allow"):
    enabled: bool = False
    recipe: Literal["nvfp4_pertoken"] = "nvfp4_pertoken"  # TE per-token activation scaling
    first_last_layers_bf16: bool = True                   # the study's f2l4
    num_first_layers_bf16: int = 2
    num_last_layers_bf16: int = 4
```

Reuse: the whole m-inf study implementation (pinned TE fork knobs,
`NVFP4_IMPLEMENTATION.md`). **Dependency risk:** main's TE pin may lack
per-token NVFP4 — the study runs on a pinned TE fork; upstreaming that pin is
part of this workstream's env story (same sqsh-bake flow as the vLLM repin).

### 2. Refit — quantize-at-export in Megatron-Bridge (new, small)

New Bridge method `AutoBridge.export_hf_weights_nvfp4`, a sibling of
`export_hf_weights_modelopt` (auto_bridge.py) that reuses its skeleton but
computes quant metadata **from the weights themselves**:

```python
def export_hf_weights_nvfp4(
    self, model, *,
    quantize_fn,                # (hf_name, weight) -> iter[(name, tensor)] : 3-tensor NVFP4 contract
    quant_patterns,             # HF-name patterns to quantize (e.g. *.experts.*)
    cpu=False, conversion_tasks=None, ...
) -> Iterable[HFWeightTuple]:
```

Reused unchanged from the modelopt exporter (all ModelOpt-free already):
conversion tasks + `build_hf_to_megatron_name_map`, TP/PP/EP gather and expert
stacking, streaming loop, ignore/pattern matching, `_nvfp4_export_names`
naming. Dropped: `collect_modelopt_quant_metadata` (quantizer buffers),
`compute_nvfp4_input_scale` and all input-scale export, qformat checks.

(Why not `export_hf_weights_quant`? Its `quant_fn` contract is a 2-tuple
`(qweight, scale)` shaped for FP8; NVFP4 needs three tensors. Extending that
contract is the fallback; a dedicated exporter is clearer.)

**The producer (`quantize_fn`)** is one pluggable function living in nemo-rl
(`nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py`, see §3):
per-expert amax → `weight_scale_2`; block-16 E4M3 scales; packed uint8 — the
same layout the probe's check B1 validated. Implementation choice, in order of
preference: vendor the ~30-line math (no fragile private import), matching
vLLM's `_quantize_moe_weight_to_nvfp4` bit-for-bit (guarded by the
cross-producer micro-test the probe README calls for). It runs on the training
GPU at export; `flashinfer.nvfp4_quantize` is an optional fast backend later.

**NeMo-RL worker hook:** `MegatronPolicyWorker` gains
`_iter_nvfp4_pertoken_refit_params()`, selected next to the existing
`_iter_params_with_optional_kv_scales` when the rollout mode is enabled.
It calls the new bridge exporter and appends kv scales exactly like today
(reuse `get_vllm_qkv_scale_names`). No changes to `MegatronQuantPolicyWorker`.

### 3. vLLM — neutral per-token W4A4 module (graduate the probe overlay)

New module `nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py`
(sibling of `quantization/fp8.py`; deliberately **not** under `nemo_rl/modelopt/`):

- The probe's `pertoken_overlay.py` classes, renamed neutrally:
  `NvFp4PerTokenConfig` / `NvFp4PerTokenFusedMoE`, registered as
  `"nvfp4_pertoken"` via `register_quantization_config`. (They still subclass
  vLLM's `ModelOptNvFp4FusedMoE` — that's vLLM's class name for the NVFP4
  checkpoint-format method, not a ModelOpt dependency.) Includes the probe's
  `.contiguous()` reload fix and the FLASHINFER_TRTLLM backend assert.
- The HF `quantization_config` override as a **literal dict** (quant_algo
  NVFP4, group_size 16, dynamic input activations, exclude_modules from the
  ignore list) — no ModelOpt `convert_hf_quant_config_format`.
- The `quantize_fn` producer from §2 (co-located so vLLM-format knowledge
  stays in one file).
- `DEFAULT_NVFP4_IGNORE` moves (or is re-exported) here; `nemo_rl/modelopt/`
  imports from the neutral module — dependency points one way only.

**Weight transport:** the fused-MoE refit machinery in #2983's
`vllm_quant_backend.py` is format-generic (suffix mapping, manifest
completeness, layerwise reload lifecycle with CUDA-graph-stable finalize,
600s ZMQ timeout, EP rejection). Factor the reusable pieces into a neutral
helper module and add a thin `NvFp4PerTokenWorkerExtension
(VllmInternalWorkerExtension)` that uses them with `require_input_scales=False`
and no `VLLM_MODELOPT_REAL_QUANT` env. The ModelOpt backend keeps working by
importing the factored helpers (mechanical refactor, no behavior change).

### 4. Config surface (v2, per config-conventions)

```python
class NvFp4PerTokenRolloutConfig(BaseModel, extra="allow"):
    enabled: bool = False
    ignore: list[str] | None = None   # None → DEFAULT_NVFP4_IGNORE
```

- Lives at `policy.generation.nvfp4_pertoken_rollout`. Defaults on the
  BaseModel; exemplar YAML documents it; recipes override minimally with
  `defaults:` inheritance.
- **Mutual exclusion, fail loudly:** startup assert that
  `nvfp4_pertoken_rollout.enabled` is not combined with the ModelOpt path's
  `generation.quant_cfg`/`real_quant`.
- **Train/rollout consistency guard** (pattern from #2983's
  `_get_real_quant_mode` cross-check): if `fp4_cfg.enabled`, warn/error when
  the layers TE keeps in BF16 (f2l4) diverge from the rollout `ignore` list.

### 5. Explicitly NOT reused (the independence claim)

`MegatronQuantPolicyWorker` and everything it exists for: `quantize_model` /
calibration data plumbing, `hide_tensor_quantizers`, quantized startup ckpt
caches, ModelOpt extra-state checkpoint handling, `VLLM_QUANT_CFG` fake-quant,
`VLLM_MODELOPT_REAL_QUANT`, `resolve_nvfp4_real_quant_mode` (that resolver
parses ModelOpt recipes; this path has no ModelOpt recipe to parse).

## Train/rollout alignment (tracked, not blocking)

Rollout is self-consistent by construction: the weights vLLM runs are exactly
the producer's output, and activations are per-token in-kernel. The residual
train/gen gap is TE's quantization recipe vs FlashInfer's (weight scale layout,
activation scale semantics, 4over6 variants). v1 accepts and *measures* it
(`gen_kl_error` / `token_mult_prob_error` — note the per-token rollout already
moved these toward BF16 in the probe); recipe alignment (TE 1D block-16
config, 4over6 parity) is a follow-up study using the m-inf harness.

## Testing

1. **Producer micro-test** (GPU unit): vendored quantize_fn ≡ vLLM's
   `_quantize_moe_weight_to_nvfp4` bitwise; determinism. (Extends probe B1.)
2. **CPU-mocked units**: exporter name/shape contract; config guards
   (mutual exclusion, ignore-list consistency); worker extension manifest
   without input scales. Mirror `test_vllm_modelopt_real_quant_config.py` fakes.
3. **Probe reuse** (GB200): checks B2/C/D run unchanged against the renamed
   registered config — same engine behavior, same thresholds
   (hybrid ≤ static×1.05; measured 1.690 vs 2.180).
4. **Functional**: 1–2-step GRPO on Qwen3-30B-A3B with fp4_cfg + pertoken
   rollout; `assert_grep` markers from the probe README; `check_metrics.py`
   gates copied from the w4a4-static recipe, tightened after green runs.
5. **Nightly GB200**: `grpo-qwen3-30ba3b-4n4g-megatron-te-nvfp4-pertoken.{yaml,sh}`.

## Phasing

- **P0** Bridge exporter + producer + micro-test (no vLLM dependency; testable
  against safetensors on any Blackwell node).
- **P1** vLLM neutral module (graduate overlay) + transport factoring + worker
  hook + config classes. Requires the vLLM repin (≥ #48538) — shared
  prerequisite with the ModelOpt per-token mode.
- **P2** e2e recipe + functional test + nightly. Requires TE pin with
  per-token NVFP4 (env-build sqsh).
- **P3** alignment study (TE↔FlashInfer recipe parity) on the m-inf harness.

## Risks

| Risk | Mitigation |
|---|---|
| TE pin on main lacks per-token NVFP4 (study uses a TE fork) | env story = same sqsh-bake as vLLM repin; P0/P1 don't need TE |
| Fused-MoE refit requires all experts local (no EP), inherited from transport | same constraint as #2983 today; TP is fine (probe: TP=2 works) |
| Refit-time quantization cost on training GPUs | producer is one batched kernel per layer (vLLM does the same at load); measure in P2, flashinfer backend if needed |
| vLLM `ModelOptNvFp4FusedMoE` base-class drift across repins | thin subclass + probe checks in CI catch it; the factored transport is ours |
| Cross-producer bit drift (vendored vs vLLM kernel) | P0 micro-test is a hard gate |
