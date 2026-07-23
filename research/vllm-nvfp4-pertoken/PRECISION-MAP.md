# NVFP4 per-token precision map — knobs, design, and current job configs

Where every NVFP4 knob lives, what precision each layer/component ends up in,
why the design keeps training and rollout aligned, and what each current job
config actually sets. Model in all cases is **Qwen3-30B-A3B-Base: 48 decoder
layers, MoE with a shared router + routed experts per layer.**

Companion docs (read for deeper history):
- `DESIGN-te-nvfp4-pertoken.md` — the per-token W4A4 architecture end to end.
- `CONFIG-fp4train.md` — the `-fp4train` training-side config reference.
- `REPLICATE-dapo-20k.md` — reproducing the DAPO-20k long run.

---

## 1. TL;DR — the precision map

Two independent pipelines (training compute vs. rollout compute) are configured
to land on the **same** NVFP4 layer set, so the training forward matches the
generation forward and `token_mult_prob_error` / `gen_kl_error` stay small.

| Component | Layers 2–43 | Layers 0–1, 44–47 (f2l4) | Set by (train) | Set by (rollout) |
|---|---|---|---|---|
| MoE **expert** linears (`fc1`/`fc2`) | **NVFP4 W4A4** | **BF16** | `te_precision` fc1/fc2 → nvfp4, minus f2l4 | `nvfp4_pertoken_rollout` (`*.experts.*`), minus f2l4 ignore |
| Attention (`qkv`, `o_proj`) | BF16 | BF16 | `te_precision` demotes `linear_qkv`/`linear_proj` | `ignore: *self_attn*` |
| MoE **router** / gate | BF16 (fp32 probs) | BF16 | `moe_router_dtype: fp32` | `ignore: *mlp.gate*` |
| Shared expert | BF16 | BF16 | not matched to nvfp4 | `ignore: *mlp.shared_expert*` |
| Norms, embeddings, `lm_head` | BF16 | BF16 | not matched to nvfp4 | `ignore: *norm* / *embed_tokens* / *lm_head*` |

**Net: only the routed MoE expert GEMMs in the interior 42 layers run in
NVFP4.** Everything else is BF16 on both sides.

"W4A4" = 4-bit **weights** (block-16 E2M1, scales fixed at refit) and 4-bit
**activations** (per-token dynamic, computed inside the FlashInfer TRT-LLM
fused-MoE kernel).

---

## 2. Two stages, one alignment goal

- **Training stage** — Megatron-Core + Transformer Engine. Knobs under
  `policy.megatron_cfg`. Consumed in `nemo_rl/models/megatron/setup.py`
  (`_setup_fp4`, ~L883–943).
- **Generation stage** — vLLM colocated rollout. Knobs under
  `policy.generation`. Consumed in `megatron_policy_worker.py` (refit filter)
  and `nemo_rl/models/generation/vllm/utils.py:resolve_generation_worker_cls`.

The two share no state. The whole design is: pick a layer set for NVFP4 in
training, then configure the rollout to quantize *exactly that same set*. If
they diverge, the rollout logprobs no longer match the training logprobs and RL
importance ratios blow up.

---

## 3. Training-stage knobs (`policy.megatron_cfg`)

NVFP4 coverage is **not one switch** — three controls compose, from coarse to
fine:

### 3.1 `fp4_cfg` — the global default
```yaml
fp4_cfg:
  enabled: true
  fp4: e2m1          # element format
  fp4_recipe: nvfp4  # scaling recipe
  fp4_param: false   # master weights stay BF16; quantize just-in-time per forward
```
Turns the **whole model NVFP4** — with only this, every TE linear in the
interior layers (attention *and* MLP) is NVFP4. `fp4_param: false` means the
stored parameter stays BF16 (optimizer updates in BF16) and is cast to FP4 on
the fly each matmul; `true` would store the param itself as FP4 at init.

### 3.2 `te_precision_config_file` — per-module override
```yaml
te_precision_config_file: examples/configs/te_precision/attn_bf16_mlp_nvfp4.yaml
```
Loaded into `TransformerConfig.quant_recipe` via
`megatron.core.quantization.utils.load_quantization_recipe`. Glob matchers
demote modules out of the global default:

| Matcher pattern | Precision |
|---|---|
| `*.linear_qkv`  | BF16 |
| `*.linear_proj` | BF16 |
| `*.linear_fc1`  | NVFP4 |
| `*.linear_fc2`  | NVFP4 |

→ **attention BF16, MLP/expert linears NVFP4.** Anything unmatched keeps the
`fp4_cfg` global default, but the router is already forced BF16/fp32 by
`moe_router_dtype: fp32`, so nothing else is silently left in FP4.
(Caveat: matchers run post-init, so `fp4_param` cannot be set here — it lives in
`fp4_cfg`.)

### 3.3 `first_last_layers_bf16` (f2l4) — whole-layer carve-out
```yaml
first_last_layers_bf16: true
num_layers_at_start_in_bf16: 2   # layers 0–1 entirely BF16
num_layers_at_end_in_bf16: 4     # layers 44–47 entirely BF16
```
Megatron `fp4_utils` keeps these boundary TransformerBlocks fully BF16
(attention + MLP). Counts are **per pipeline stage**. Boundary layers carry the
most FP4-sensitive signal, so they stay high precision.

### 3.4 NVTE numerics env vars (shape the FP4 math, not the layer set)
```yaml
env_vars:
  NVTE_NVFP4_ROW_SCALED_ACTIVATION: "1"    # per-token (row) activation scaling — the "per-token" in the name
  NVTE_NVFP4_DISABLE_RHT: "1"              # no random Hadamard transform
  NVTE_NVFP4_DISABLE_2D_QUANTIZATION: "1"  # 1D block scaling only
  NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING: "1"  # deterministic RNE
  NVTE_BACKWARD_OVERRIDE: "dequantized"    # backward runs the dequantized (BF16) path
```
Also `optimizer.use_precision_aware_optimizer: false`.

---

## 4. Generation-stage knobs (`policy.generation`)

```yaml
nvfp4_pertoken_rollout:
  enabled: true
  # quant_patterns defaults to ["*.experts.*"] — MoE experts only (kernel is MoE-only)
  ignore:
    - "*lm_head*"
    - "*mlp.gate"                 # exact router prefix; a trailing ".*" would NOT match it
    - "*mlp.gate.*"
    - "*mlp.shared_expert*"
    - "*self_attn*"
    - "*embed_tokens*"
    - "*input_layernorm*"
    - "*post_attention_layernorm*"
    - "*norm*"
    - "*.layers.0.mlp.experts*"   # f2l4 — mirror the training BF16 boundary layers
    - "*.layers.1.mlp.experts*"
    - "*.layers.44.mlp.experts*"
    - "*.layers.45.mlp.experts*"
    - "*.layers.46.mlp.experts*"
    - "*.layers.47.mlp.experts*"
```

- **`enabled`** selects the `NvFp4PerToken*GenerationWorker` (see
  `resolve_generation_worker_cls`) and wraps the refit weight stream so matching
  weights are real-quantized to NVFP4 on the way into vLLM. Activations are
  quantized per-token inside the fused-MoE kernel → W4A4. At runtime this
  resolves to a vLLM `quantization_config` with
  `weights {dynamic: false, num_bits: 4, group_size: 16}` and
  `input_activations {dynamic: true, num_bits: 4, group_size: 16}`.
- **`quant_patterns`** (default `["*.experts.*"]`) — allowlist of what gets
  quantized at refit. MoE experts only.
- **`ignore`** — patterns kept BF16 during rollout. **Gotcha: an explicit
  `ignore` REPLACES the built-in `DEFAULT_NVFP4_IGNORE`, it does not merge**
  (`NvFp4PerTokenRolloutConfig.resolved_ignore`). That is why every recipe
  repeats the default nine entries and then appends the six f2l4 layer excludes.

Related rollout knobs used by these recipes: `enforce_eager: false` (CUDA
graphs, ~10× faster on this path), `moe_backend` **unset** (vLLM auto-selects
`flashinfer_trtllm` for the quantized experts; `triton` is invalid for NVFP4
MoE and is only injected by the launcher for `PRECISION=bf16`),
`gpu_memory_utilization: 0.5`, vLLM `tp1`.

---

## 5. How the carve-outs compose, layer by layer

Start from "everything NVFP4" (`fp4_cfg`), then subtract:

```
fp4_cfg.enabled           →  ALL interior linears NVFP4  (attention + MLP)
  − te_precision matchers  →  attention (qkv/proj) back to BF16   ── leaves MLP/expert fc1/fc2 NVFP4
  − first_last_layers_bf16 →  layers 0–1, 44–47 fully BF16         ── leaves experts NVFP4 in layers 2–43
  (router/shared/norms/embed already BF16 via moe_router_dtype / no match)
= NVFP4 only on routed-expert fc1/fc2 in layers 2–43
```

The rollout `ignore` list is the mirror image: it starts from
`quant_patterns=*.experts.*` (all expert projections) and subtracts the same
f2l4 layers plus everything non-expert.

### Alignment guard
`megatron_policy_worker._warn_fp4_f2l4_rollout_ignore_mismatch` computes the
f2l4 BF16 layer set from `num_layers_at_start/end_in_bf16` and warns if the
rollout `ignore` patterns fail to exclude those layers' experts — i.e. it
catches exactly the drift where a layer trains BF16 but the rollout would
quantize it. It is a **warning, not a hard error**: if you change
`num_layers_at_end_in_bf16` you must hand-edit the `ignore` list too, since the
two are maintained separately.

---

## 6. Current NVFP4 job configs

All long-run DAPO configs inherit down a single chain; only the quant deltas
differ. Base → `…-bf16.yaml` (no quant) → `…-nvfp4-pertoken.yaml` (all quant
lives here) → variants.

| Recipe | Inherits | Quant vs. `-pertoken` | Purpose / notes |
|---|---|---|---|
| `…-nvfp4-pertoken.yaml` | `…-bf16` | **defines the full W4A4 layer set** (§3–4) | The canonical NVFP4 long run. Everything above is set here. |
| `…-nvfp4-pertoken-cudagraphs.yaml` | `…-pertoken` | identical | CUDA-graph probe (`enforce_eager: false`), short, no ckpt. |
| `…-nvfp4-pertoken-r3.yaml` | `…-pertoken` | identical layer set | Adds **router replay** + the packing/sampling changes it needs (dynamic sampling off, overlong filter off, `batch_multiplier: 1`, sequence packing on, dynamic batching off, prefix caching off). Quant unchanged. |
| `…-nvfp4-pertoken-r3-probe.yaml` | `…-r3` | identical | 3-step de-risk of r3 (no val/ckpt). |
| `grpo-qwen3-30ba3b-4n4g-megatron-nvfp4-pertoken-fp4train.yaml` | 4n4g M1 pertoken | same 3-layer quant scheme | Smaller-topology training-side reference; also pins `moe_router_dtype: fp32` (fp64 router probs die under FP4 expert padding). |

### The live run: `…-nvfp4-pertoken-r2-20260721`
Job **2422244–2422249**, wandb `054c75d9…`, started 2026-07-21, live at ~step
419/800 as of writing. **There is no committed `-r2` recipe** — it was launched
straight from `…-nvfp4-pertoken.yaml` with only path/cluster/wandb-name
overrides. `-r2` is a run-name tag, not a config variant.

Verified from the resolved `MasterConfig` in `2422244-logs/ray-driver.log`:
- Training: `fp4_cfg {enabled, e2m1, nvfp4, fp4_param:false}`, `first_last_layers_bf16:true`, start=2/end=4, `te_precision=attn_bf16_mlp_nvfp4.yaml`, all NVTE_* vars, `use_precision_aware_optimizer:false`.
- Rollout: `NvFp4PerTokenGenerationWorker` on every rank; resolved vLLM
  `quantization_config` = NVFP4 W4A4 (weights static block-16, activations
  per-token dynamic) with the full f2l4 `ignore`. No f2l4 mismatch warning fired.

**What it is NOT:** no router replay (that's r3-only), and — because it's the
plain pertoken config — DAPO sampling is fully on (`use_dynamic_sampling:true`,
`overlong_filtering:true`, `batch_multiplier:3`, `dynamic_batching:true`,
`sequence_packing:false`, prefix caching on). So r2-20260721 is the **NVFP4
W4A4 long run without router replay**, on full DAPO dynamic sampling — the
clean baseline to diff against the r3 (router-replay) runs, since the
quantization layer set is identical between them.

---

## 7. One-line mental model

> Global FP4 default, minus attention (TE recipe), minus the first-2/last-4
> layers (f2l4) → NVFP4 lives only on routed-expert GEMMs in layers 2–43, and
> the rollout `ignore` list is hand-kept to match that exact set.
