# NVFP4 per-token training (M2) — config reference

How the `-fp4train` recipe configures real NVFP4 Megatron/TE training so it
matches the per-token W4A4 vLLM rollout. Adapted from the rl-fp4 m-inf
precision study (`run_qwen30ba3b_study_m_inf_fa4_precision.sh`).

## Coverage is set in two composing layers

NVFP4 module coverage is NOT one switch. Two independent controls compose:

1. **`fp4_cfg` — global default precision.** Turns the whole model NVFP4.
   ```yaml
   policy.megatron_cfg.fp4_cfg:
     enabled: true
     fp4: e2m1          # element format
     fp4_recipe: nvfp4  # scaling recipe
     fp4_param: false   # keep master weights high-precision (cast activations only)
   ```
   With only this, EVERY TE linear in the interior layers is NVFP4 —
   attention (`linear_qkv`, `linear_proj`) AND MLP (`linear_fc1`,
   `linear_fc2`).

2. **`te_precision_config_file` — per-module overrides.** A TE precision
   recipe (`TransformerConfig.quant_recipe`, loaded by
   `megatron.core.quantization.utils.load_quantization_recipe`) demotes
   specific modules out of the global default via glob matchers.
   ```yaml
   policy.megatron_cfg.te_precision_config_file: examples/configs/te_precision/attn_bf16_mlp_nvfp4.yaml
   ```
   `attn_bf16_mlp_nvfp4.yaml` matches `*.linear_qkv` / `*.linear_proj` -> BF16
   and `*.linear_fc1` / `*.linear_fc2` -> NVFP4, i.e. **attention BF16,
   MLP/experts NVFP4**. (Study alternative `qkv_oproj_mxfp8_fc_nvfp4.yaml`
   uses MXFP8 attention instead of BF16.)

Third, coarsest control:

3. **`first_last_layers_bf16` (f2l4).** Boundary TransformerBlocks revert to
   BF16 entirely (attention + MLP).
   ```yaml
   first_last_layers_bf16: true
   num_layers_at_start_in_bf16: 2
   num_layers_at_end_in_bf16: 4   # of 48 -> 42 interior layers quantized
   ```

Net for the current `-fp4train` recipe: layers 0-1 and 44-47 fully BF16; in
the 42 interior layers, attention BF16 and MoE experts (`linear_fc1/fc2`)
NVFP4.

## Why attention-BF16 (the train/rollout alignment)

`gen_kl_error` / `token_mult_prob_error` compare the TRAINING-forward
logprobs against the vLLM ROLLOUT logprobs (`grpo.py:2199`). They shrink when
the two forwards are the SAME function, not merely when both are quantized.
The per-token rollout quantizes ONLY MoE experts (`DEFAULT_NVFP4_IGNORE`
keeps `self_attn`, router, shared experts, norms in BF16). So:

- Global `fp4_cfg` alone quantizes training attention -> diverges from the
  BF16-rollout attention -> inflated gen_kl (M2 job 2411336: gen_kl 0.037,
  token_mult_prob 15.8->9.9 vs BF16-train M1's 0.018 / ~1.1).
- Adding `attn_bf16_mlp_nvfp4.yaml` keeps training attention BF16, matching
  the rollout. MEASURED (job 2411458): gen_kl 0.0113, js 0.0029,
  token_mult_prob 2.97 — BELOW both the all-NVFP4 run AND BF16-train M1,
  because this config matches the rollout on both axes (attention BF16 both
  sides, experts NVFP4 both sides; M1 trained experts BF16 vs NVFP4 rollout).
  Driver gates tightened back to 0.03/0.007 (this config clears with ~2.5x
  margin). NOTE token_mult_prob_error spiked to 180 at STEP 1 (one outlier
  sequence; exp-of-logprob-diff is fragile) then settled to 2.97 at step 2 —
  gen_kl/js (mean-based) are the reliable signals; step-1 tmpe is noisy.

Even matched coverage won't drive the gap to zero: TE row/per-token-scaled
NVFP4 (training) and flashinfer block-16 NVFP4 (rollout) are different
kernels, so the expert-GEMM errors don't fully cancel (residual gen_kl 0.011).

## NVTE activation-quantization knobs (env_vars)

```yaml
policy.megatron_cfg.env_vars:
  NVTE_NVFP4_ROW_SCALED_ACTIVATION: "1"     # per-token (1D row) activation scale — the point of M2
  NVTE_NVFP4_DISABLE_RHT: "1"               # no random Hadamard transform
  NVTE_NVFP4_DISABLE_2D_QUANTIZATION: "1"   # 1D (per-token), not 2D block
  NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING: "1"
  NVTE_BACKWARD_OVERRIDE: "dequantized"     # forward NVFP4, backward dequantized (stability)
  # optional: NVTE_NVFP4_4OVER6: "activations"  # 4-of-6 mantissa mode on activations
```
Requires TE >= 937c4de (the ROW_SCALED knob; absent from release_v2.15) —
pinned for the mcore extra in `pyproject.toml`, baked into the v4 image.

## Other required overrides

- `moe_router_dtype: fp32` — fp64 router probs (perf parent's default) hit
  TE's `fused_multi_row_padding`, which has no double kernel
  (`GetTransformerEngineDType: Invalid type (7)`). Must be fp32 (or bf16).
- `optimizer.use_precision_aware_optimizer: false`.
- Rollout `nvfp4_pertoken_rollout.ignore` must also exclude the f2l4 layers'
  experts, or those layers train BF16 while the rollout quantizes them
  (the worker warns; refit drops from 48 to 42 quantized layers).

## Startup markers (driver greps)

- `[fp4_cfg] Megatron FP4 training enabled: fp4=e2m1 recipe=nvfp4 fp4_param=False`
- `[fp4_cfg] TE per-module precision recipe loaded from <path>`
- `[nvfp4_pertoken] per-token NVFP4 activation scaling active` (vLLM side)
- `[nvfp4_pertoken] refit: quantized 42 expert layers ... passthrough 2739`
