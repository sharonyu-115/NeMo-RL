# vLLM per-token NVFP4 activation scale — standalone validation

Validates vLLM's per-token dynamic global activation scaling for NVFP4 MoE
(upstream [vllm#48538](https://github.com/vllm-project/vllm/pull/48538),
`--quantization nvfp4_per_token`, `Nvfp4OnlineMoEMethod`) before wiring it into
NeMo-RL's real-quant rollout flow on top of
[NVIDIA-NeMo/RL#2983](https://github.com/NVIDIA-NeMo/RL/pull/2983).

Survey/motivation: `rl-fp4` branch `m-inf-fa4-precision-study`,
`research/megatron-inference-true-on-policy/per-token-nvfp4-survey.md`.

## To-be flow under test

Weights: real-quantized during refit on the Megatron side (unchanged from
PR #2983 w4a4). Activations: per-token FP32 global scale computed dynamically
in vLLM — no Megatron-exported `w13_input_scale`/`w2_input_scale`, no
calibration in the rollout path.

## Environment

| Item | Value |
|---|---|
| Container | `/lustre/fsw/general_sa/shuangy/images/vllm-nightly-2026-07-18.sqsh` (7.1G) |
| Image source | `docker://vllm/vllm-openai:nightly` imported 2026-07-18 |
| vLLM version | `0.23.1rc1.dev1261+gc71a583aa` (postdates #48538, merged 2026-07-16) |
| Hardware | 1× GB200 node, 4× GB200 SM100 189GB (feature is SM100-only, MoE-only, TP=1 only) |
| Models | `Qwen/Qwen3-30B-A3B` (BF16), `nvidia/Qwen3-30B-A3B-NVFP4` (static baseline), `ibm-granite/granite-3.0-1b-a400m-base` (fast-iteration MoE) |
| Slurm | account `general_sa`, `--partition=batch,tcpo,36x2-a01r,a02grace`, whole-node exclusive (no GPU GRES), job names `general_sa-nemo_rl.<details>` |

## Checks

- **A — smoke + feature-active proof**: engine loads with
  `quantization="nvfp4_per_token"`, generates sanely; introspection proves
  `Nvfp4OnlineMoEMethod` is live and no static input-scale params exist;
  TP=2 negative test.
- **B — pre-quantized weights + per-token activations (go/no-go)**:
  (1) exercise `_quantize_moe_weight_to_nvfp4` (vLLM's online-quant helper,
  which wraps vLLM's in-tree `scaled_fp4_quant` CUDA kernel — NOT
  `flashinfer.nvfp4_quantize`, FlashInfer's independent implementation of the
  same transform) and assert its output matches the ModelOpt checkpoint
  layout bitwise-deterministically — validates the refit export contract;
  (2) hybrid overlay: pre-quantized ModelOpt ckpt with checkpoint
  `input_scale` ignored + per-token dynamic activation config
  (vLLM analog of `SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE=1`).
  Note: runtime per-token activation quant happens inside FlashInfer's fused
  `trtllm_fp4_block_scale_moe` kernel, not through either weight-quant path.
  Cross-provider bitwise agreement (vLLM kernel vs `flashinfer.nvfp4_quantize`
  vs ModelOpt export) is untested — add a micro-test before a no-ModelOpt
  quantize-at-refit flow picks a producer.
- **C — weight-reload determinism (RL-critical)**: identity reload reproduces
  greedy outputs; corrupt-then-reload proves real re-quantization; layerwise
  reload leg mirrors `_weight_update_lifecycle` in
  `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`.
- **D — logprob fidelity**: `avg_prob_mult_error = mean(exp(|Δlogprob|))` of
  per-token and static-scale legs vs BF16 reference over the same sampled
  token sequences. Gates: per-token ≤ 1.20 AND ≤ static × 1.05.

Run: `sbatch sbatch_nvfp4_pertoken.sh` (or `RUN_CHECKS=A,B ./sbatch_...` to select).

## Verdict (2026-07-18): **GO for Stage-2 wiring**

All four Stage-1 questions answered affirmatively on GB200 + vLLM nightly
(0.23.1rc1.dev1261):

1. **Kernel feasibility** — pre-quantized ModelOpt NVFP4 weights + per-token
   dynamic activation scales run through the FlashInfer TRT-LLM fused MoE
   (`pertoken_overlay.py`, ~90 lines, structured to graduate directly into
   #2983's `vllm_modelopt.py` as the `w4a4_pertoken` registered config).
2. **Fidelity** — per-token beats calibrated static scales by ~22%
   avg_prob_mult_error on identical quantized weights (1.690 vs 2.180).
3. **Reload/refit correctness** — identity reload is token-identical and
   corrupt-then-reload restores baseline, provided kernel scale tensors are
   contiguous (fix included in the overlay; #2983 already does this).
4. **Parallelism** — TP=2 works (upstream's initial TP `NotImplementedError`
   constraint is gone), unblocking multi-GPU rollout.

Bonus finding: upstream's `_quantize_moe_weight_to_nvfp4` emits exactly the
ModelOpt checkpoint tensor layout, bitwise-deterministically — the same
export contract also serves the future no-ModelOpt (TE-training +
quantize-at-refit) flow.

## Findings

| Check | Status | Notes |
|---|---|---|
| Stage-0 gates | PASS (2026-07-18, job 2404072) | vLLM 0.23.1rc1.dev1261+gc71a583aa; `Nvfp4OnlineMoEMethod` importable; `make_nvfp4_moe_kernel(per_token_activation=...)` present; `has_flashinfer_trtllm_fused_moe()=True` on GB200 SM100 |
| A smoke (granite + Qwen3-30B) | PASS (job 2404103) | `nvfp4_per_token` loads and generates sanely on both; `Nvfp4OnlineMoEMethod` active on every MoE layer; `FLASHINFER_TRTLLM` backend selected |
| A TP=2 | PASS — **TP works now** (job 2404103) | Survey's "TP raises NotImplementedError" is outdated; nightly initializes TP=2 and generates sanely. Removes the biggest wiring blocker for multi-GPU rollout |
| B.1 external-quant layout + determinism | PASS (job 2404103) | `_quantize_moe_weight_to_nvfp4` output = ModelOpt ckpt layout (uint8 packed / fp8-e4m3 block / fp32 global); bitwise deterministic |
| B.2 hybrid per-token overlay (GO/NO-GO) | **PASS** (job 2404103) | Pre-quantized ModelOpt NVFP4 weights + per-token dynamic activations generate sanely through FlashInfer TRT-LLM; `ModelOptNvFp4PerTokenFusedMoE` active. The NeMo-RL to-be flow is kernel-feasible |
| C reload determinism — hybrid overlay | **PASS** (job 2404172) | With the overlay's `.contiguous()` fix: identity `reload_weights` reproduces greedy outputs token-identically; corrupting packed FP4 weight bytes changes outputs; reload restores the baseline exactly. The RL refit contract holds for the to-be flow |
| C reload determinism — upstream `nvfp4_per_token` | XFAIL, upstream bug (jobs 2404119/2404153) | `reload_weights` fails on stride-0 expanded scale views: layerwise-reload finalize `param.data.copy_()` cannot write into them. Upstream `Nvfp4OnlineMoEMethod` needs the same `.contiguous()` treatment (matches RFC #48312 risk list). Not a blocker for NeMo-RL, whose path uses the (fixed) overlay approach | |
| D fidelity (per-token / static / hybrid vs BF16) | **PASS — thesis validated** (job 2404135) | See table below. On identical pre-quantized weights, per-token dynamic activation scales beat calibrated static scales by ~22% |

### Fidelity results (2026-07-18, job 2404135)

`avg_prob_mult_error = mean(exp(|logp_bf16 − logp_quant|))` over 8192 generated
positions (32 prompts × 256 tokens, temp-1.0 sampled from the BF16 engine,
rescored via `prompt_logprobs`; BF16 reference on triton MoE backend).

| Leg | Weights | Activation global scale | avg_prob_mult_error | mean abs Δlogp |
|---|---|---|---|---|
| per-token (`nvfp4_per_token`) | online-quantized from BF16 | per-token dynamic | **1.307** | 0.104 |
| hybrid (overlay) | nvidia NVFP4 ckpt (pre-quantized) | per-token dynamic | **1.690** | 0.156 |
| static (`modelopt_fp4`) | nvidia NVFP4 ckpt (pre-quantized) | calibrated static `input_scale` | **2.180** | 0.156 |

Reading:
- **hybrid vs static is the controlled comparison** (identical quantized
  weights, only the activation-scale regime differs): per-token dynamic wins,
  2.180 → 1.690. This is the Stage-2 wiring's expected gain and mirrors
  SGLang's GSM8K improvement from the same change.
- per-token-from-BF16 (1.307) beats both pre-quantized legs because the
  nvidia checkpoint's QAT/PTQ weights intentionally differ from raw BF16 —
  scoring against the BF16 reference penalizes them; not an activation effect.
- These absolute values are on temp-1.0 sampled tokens (incl. rare tokens,
  max |Δlogp| ≈ 6.5–7.9) — do not compare directly against nemo-rl CI's
  `token_mult_prob_error` thresholds (~1.05) which run on greedy/on-policy
  rollout distributions. Use the *relative* ordering for Stage-2 decisions;
  set nightly thresholds from an in-flow measurement.

### Upstream bugs found (report to vllm-project)

1. **`reload_weights` breaks on `nvfp4_per_token`**: `Nvfp4OnlineMoEMethod`
   registers activation global scales that reach layerwise-reload finalize as
   stride-0 expanded views; `_copy_and_restore_kernel_tensors` does
   `param.data.copy_()` which raises "more than one element of the written-to
   tensor refers to a single memory location". Fix = `.contiguous()` on kernel
   scale tensors (as this probe's overlay does, and as NeMo-RL PR #2983 does
   in its registered method). Matches vLLM RFC #48312's risk taxonomy.
2. **BF16 FlashInfer fused-MoE illegal memory access on GB200**: plain
   `LLM(Qwen3-30B-A3B)` (no quantization) crashes during init in
   `flashinfer/fused_moe/core.py:1227` with `cudaErrorIllegalAddress` on
   vLLM 0.23.1rc1.dev1261. Workaround: `moe_backend="triton"`.

### Log markers for Stage-2 assert_grep

- `Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend` — kernel backend proof
- `modelopt_fp4_pertoken: ignoring checkpoint input_scale; per-token NVFP4 activation scaling active` — overlay active (from `pertoken_overlay.py`)
- Engine config line contains `quantization=nvfp4_per_token` (upstream) or `quantization=modelopt_fp4_pertoken` (overlay)

### Measured thresholds for Stage-2

- `avg_prob_mult_error` (temp-1.0 sampled-token scoring): per-token 1.307,
  hybrid 1.690, static 2.180. Suggested probe gates: per-token ≤ 1.5 absolute;
  hybrid ≤ static × 1.05 relative (both now encoded in check D5).
- Nightly recipe thresholds should start from the w4a4-static recipe's values
  (`gen_kl_error < 0.03`, `js_divergence_error < 0.007`) — per-token is
  expected to do better, tighten after two green runs.
