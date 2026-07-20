# Design: TE NVFP4 per-token W4A4 — train → refit → vLLM rollout (no ModelOpt)

Status: **as-built** (rev. 2026-07-19; original draft same date). Companion to
this directory's probe (`README.md`) and the execution ledger (`PROGRESS.md`).
P0+P1 are implemented and GB200-validated (commits aab14dd7f, fe56561bb);
sections below describe the implemented architecture, with "superseded" notes
where implementation diverged from the original draft and why.

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
 2. REFIT  iter_nvfp4_pertoken_weights(<existing HF export stream>)
             per-expert HF names, quantized on-GPU pre-IPC:
             ├ .weight          packed FP4 uint8
             ├ .weight_scale    block-16 E4M3
             └ .weight_scale_2  FP32 (amax/(448·6), from master weight)
           — no input scales; draft weights + kv scales pass through —
           → ZMQ IPC / collective, layerwise reload lifecycle
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

### 2. Refit — quantize in-flight in nemo-rl (as built)

**Superseded: the original draft added a Megatron-Bridge exporter
(`AutoBridge.export_hf_weights_nvfp4`).** Not needed: the base worker already
receives TP-gathered, HF-named, on-GPU tensors from
`megatron_bridge.export_hf_weights`, so quantization is an **iterator filter
inside nemo-rl** — same compute point (post-gather, pre-IPC ⇒ 4x transfer
saving preserved), no cross-repo PR, no submodule bump. The Bridge exporter
remains an optional later streamlining if refit profiling shows per-expert
kernel-launch overhead (batch into stacked (E,N,K) calls bridge-side).

As built, both pieces live in the vLLM-free module
`nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py`:

- **Producer** `quantize_nvfp4_weight(weight)` — pure torch (no vLLM import:
  it runs on training workers whose mcore venv has no vLLM), 2D per-tensor or
  3D stacked-expert; per-expert amax → fp32 `weight_scale_2`; dynamic block-16
  e4m3 scales; RNE-on-grid E2M1 rounding; nibble packing. Verified
  **bit-identical** to vLLM's `_quantize_moe_weight_to_nvfp4` on GB200.
- **Refit filter** `iter_nvfp4_pertoken_weights(base_iter, quant_patterns,
  ignore_patterns)` — quantizes matching `*.weight` entries of the export
  stream into `.weight/.weight_scale/.weight_scale_2` (the ModelOpt NVFP4 HF
  checkpoint layout, per-expert names); everything else passes through.

**Worker hook (as built):** rather than a parallel iterator, the existing
`MegatronPolicyWorker._iter_params_with_optional_kv_scales` was renamed to
`..._impl` and re-exposed as a wrapper that applies the filter when
`generation.nvfp4_pertoken_rollout.enabled`. Because prepare_refit_info
metadata, ZMQ streaming, and collective broadcast all consume this single
iterator, the advertised state-dict info and streamed payloads stay consistent
by construction (no separate `build_nvfp4_refit_state_dict_info` needed, as
the draft had assumed). No changes to `MegatronQuantPolicyWorker`.

### 3. vLLM — per-token W4A4 modules (as built)

**Superseded: "one neutral module" split into three** — module-scope vLLM
subclasses (required so the pickled quantization config can be re-imported by
vLLM's EngineCore subprocess) contradict training-venv importability, so:

- `quantization/nvfp4_pertoken.py` — vLLM-free: producer, refit filter,
  `DEFAULT_NVFP4_IGNORE`, `NvFp4PerTokenRolloutConfig`,
  `build_nvfp4_pertoken_hf_quant_config`. Importable on training workers.
- `quantization/nvfp4_pertoken_vllm.py` — vLLM-side:
  `ModelOptNvFp4PerTokenFusedMoE`, `NvFp4PerTokenConfig`,
  `register_nvfp4_pertoken()`, `NvFp4PerTokenWorkerExtension`,
  `configure_nvfp4_pertoken_engine_kwargs()`.
- `quantization/nvfp4_pertoken_worker.py` — Ray generation workers
  (sync/async) that inject the engine kwargs at `_create_engine`; selected by
  `resolve_generation_worker_cls`.

**Superseded: "renamed neutrally".** The FusedMoE method class MUST keep the
"ModelOpt" substring: vLLM's `RoutedExperts.weight_loader` duck-types NVFP4
expert-scale loading on `"ModelOpt" in quant_method.__class__.__name__`
(routed_experts.py) — a neutral rename silently breaks initial load. The
registered method name is still `nvfp4_pertoken`.

The HF `quantization_config` override is a literal dict that mirrors the real
ModelOpt NVFP4 checkpoint schema **key-for-key** (`ignore`, `targets`,
`producer`; the parser is shape-sensitive), with the single per-token delta
`input_activations.dynamic=true`. Engine init uses `load_format="dummy"`: the
BF16 training checkpoint cannot fill NVFP4-shaped params, and the first refit
(which always precedes the first generation) supplies every weight.

**Superseded: transport factoring from #2983 — not needed at all.** Because
the refit stream uses per-expert checkpoint-layout names, vLLM's native
loaders handle everything; no fused-family suffix mapping or manifest
completeness code exists in this path. `NvFp4PerTokenWorkerExtension` is ~40
lines over the base lifecycle hooks: initialize/finalize layerwise reload
around each update (per-token kernel rebuilt into CUDA-graph-stable storage),
fatal refit errors, accelerator fence before IPC ack.

### 4. Config surface (v2, per config-conventions)

```python
class NvFp4PerTokenRolloutConfig(BaseModel, extra="allow"):
    enabled: bool = False
    ignore: list[str] | None = None          # None → DEFAULT_NVFP4_IGNORE
    quant_patterns: list[str] = ["*.experts.*"]  # refit-quantized allowlist (MoE-only kernel)
```

- Lives at `policy.generation.nvfp4_pertoken_rollout` (NotRequired key on the
  `VllmConfig` TypedDict; defaults on the BaseModel). Exemplar-YAML
  documentation deferred to the recipe PR to avoid reference_configs churn.
- **Mutual exclusion (implemented):** `resolve_generation_worker_cls` raises
  `ValueError` when combined with the ModelOpt path's
  `generation.quant_cfg`/`real_quant`.
- **Train/rollout consistency guard (deferred to PR-2):** when `fp4_cfg`
  lands, warn/error when the layers TE keeps in BF16 (f2l4) diverge from the
  rollout `ignore` list.

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
