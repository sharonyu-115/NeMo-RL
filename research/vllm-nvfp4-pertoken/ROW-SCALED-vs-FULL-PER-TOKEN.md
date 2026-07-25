# Row-scaled vs full per-token NVFP4 forward (and why it gates backward)

Analysis of the two "per-token" NVFP4 forward modes in the TE fork
`cael-ling/TransformerEngine@690ffea4` (PR #3045, TE `2.18.0.dev0` line), and why
enabling NVFP4 per-token **backward** necessarily swaps the forward kernel.

Source paths below are relative to the TE checkout
(`transformer_engine/…`), verified against `690ffea4`.

## Shared substrate: the NVFP4 number format

Both modes store values as **E2M1** with **two levels of scale** — this part is
identical:

- **L1 (inner):** an **E4M3** scale per **16-element micro-block**.
- **L2 (outer):** an **FP32** scale.

The modes differ only in *how the L2 outer scale is computed*, *in which
directions the cast runs*, and *which kernel path is taken*. (`common/recipe/__init__.py:487`
`NVFP4BlockScaling` docstring describes the 2-level block scaling.)

## Mode A — row-scaled activation (today's `-fp4train` forward)

Enabled by `NVTE_NVFP4_ROW_SCALED_ACTIVATION=1`
(`NVFP4BlockScaling.row_scaled_activation`, `common/recipe/__init__.py:583`).

- It is a **flag on the standard per-tensor NVFP4 cast**, not a separate kernel
  (`NVFP4Quantizer.row_scaled_nvfp4`, `pytorch/tensor/nvfp4_tensor.py:131`).
- The only change: emitted **forward-activation** tensors store **one FP32 outer
  amax per row instead of one per tensor**
  (`nvfp4_tensor.py:130` — *"emitted NVFP4 tensors store one FP32 amax per row"*;
  recipe:531 — *"forward activation quantizers emit row-scaled NVFP4 … rowwise
  amax stored as a vector, one FP32 value per row"*).
- A **row = one token's activation vector**, so per-row outer scale = per-token
  forward scale. This is what reproduces vLLM's per-token W4A4 activation scaling
  on the training forward.
- **Scope = forward activations only.** Weights keep a per-tensor outer scale; the
  backward direction is untouched. `-fp4train` pairs it with
  `NVTE_BACKWARD_OVERRIDE=dequantized`, so the backward GEMM runs in BF16
  (operands dequantized). It is a minimal, activation-only overlay.

## Mode B — full per-token (`NVTE_NVFP4_PER_TOKEN=1`)

Enabled by `NVTE_NVFP4_PER_TOKEN=1`
(`NVFP4BlockScaling.nvfp4_per_token()`, `common/recipe/__init__.py:597`), or by
constructing `NVFP4PerTokenBlockScaling`.

- It selects a **distinct cast + fused-GEMM path** — *"Per-token NVFP4 cast
  (**replaces** the per-tensor 1A/2A paths)"* (`nvfp4_tensor.py:144`). A different
  kernel, not a flag on the old one.
- Per-row/per-token outer amax is **intrinsic** to this cast
  (`nvfp4_tensor.py:200` — *"NVFP4 per-token already encodes per-row outer
  amax"*). That is why `row_scaled_nvfp4` is **forbidden / redundant** here (the
  two are mutually exclusive) and is force-disabled.
- The per-token cast produces the operand in **both directions** — rowwise for
  the forward and **columnwise/transpose for the backward**. That columnwise
  per-token operand is what makes **FP4 dgrad/wgrad possible**.
- It also rewrites the surrounding recipe via `_force_per_token_settings`
  (`common/recipe/__init__.py:620`): 2D weight scaling and 4over6 hard-off,
  `row_scaled_activation=False`, and RHT/SR become **opt-in**
  (`NVTE_NVFP4_PER_TOKEN_{RHT,SR}=1`).

## Side-by-side

| | Row-scaled (today) | Full per-token |
|---|---|---|
| Kernel | standard NVFP4 cast + `row_scaled` flag | dedicated per-token cast (replaces 1A/2A) |
| L2 outer scale | per-**tensor**, except per-**row** on fwd activations | per-**row/token**, intrinsic |
| Directions cast | forward (rowwise) only | rowwise **and** columnwise |
| Enables FP4 backward? | **No** (only rowwise exists) | **Yes** (columnwise operand produced) |
| Weight 2D / 4over6 | independent (`-fp4train` disables 2D) | forced off |
| RHT / SR | via `NVTE_NVFP4_DISABLE_*` | opt-in (`per_token_{rht,sr}`) |

## Why backward forces the forward to change

FP4 dgrad/wgrad needs the **columnwise (transpose) per-token-scaled operand**.
The row-scaled overlay only ever produces the rowwise forward activation
scaling, so it structurally cannot feed a quantized backward. Turning on
per-token backward therefore *must* switch to the per-token cast, which also
force-disables row-scaling (`_force_per_token_settings`). The gate itself is
`backward_override`: FP4 backward runs only when it is `None`
(`pytorch/module/base.py:1602` — `use_fp8_bwd = ctx.fp8 and ctx.backward_override is None`;
consumed in `pytorch/module/layernorm_linear.py:214,228,320,450`).

## Consequence for the A/B (3 legs, not 2)

Because switching on per-token backward *also* swaps the forward cast kernel, the
forward activation numerics change before any gradient is quantized. A clean
comparison needs three legs:

1. **row-scaled fwd + dequant bwd** — today's `-fp4train` baseline.
2. **per-token fwd + dequant bwd** — isolates the forward-kernel swap
   (`NVTE_NVFP4_PER_TOKEN=1`, backward still `dequantized`).
3. **per-token fwd + FP4 bwd** — adds the backward
   (`NVTE_NVFP4_PER_TOKEN=1`, `NVTE_BACKWARD_OVERRIDE` unset).

Leg (1)→(2) measures how much the forward-cast swap alone moves
`token_mult_prob_error` / `gen_kl`; leg (2)→(3) isolates the backward's effect on
convergence / `grad_norm` / loss. The exact per-token-vs-row-scaled forward delta
is a kernel-level numeric difference — measure it, don't predict it.

## NeMo-RL wiring (this campaign)

The typed field `policy.megatron_cfg.fp4_cfg.backward` drives all three legs
(`nemo_rl/models/megatron/fp4_env.py`):
`dequantized` → `NVTE_BACKWARD_OVERRIDE=dequantized`;
`nvfp4_pertoken` → `NVTE_NVFP4_PER_TOKEN=1` + clear `NVTE_BACKWARD_OVERRIDE`.
Leg (2) is `backward` left at `dequantized` **plus** `NVTE_NVFP4_PER_TOKEN=1` in
`env_vars`; leg (3) is `backward: nvfp4_pertoken`.

### Per-token runtime constraints (must hold or TE raises)

Per `nvfp4_per_token()`'s docstring, the per-token path does **not** support
`fuse_wgrad_accumulation`, sequence-parallel single-direction cast, comm-overlap,
or output-quant — these raise at runtime. `-fp4train` already disables the
relevant knobs (`gradient_accumulation_fusion: False`, `sequence_parallel: False`,
no TP comm-overlap); any 8n4g port must keep the same guards.
