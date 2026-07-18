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
  (1) quantize BF16 expert weights outside vLLM with `flashinfer.nvfp4_quantize`
  and assert numerics identical to vLLM's own online quant — validates the
  refit export contract; (2) hybrid overlay: pre-quantized ModelOpt ckpt with
  checkpoint `input_scale` ignored + per-token dynamic activation config
  (vLLM analog of `SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE=1`).
- **C — weight-reload determinism (RL-critical)**: identity reload reproduces
  greedy outputs; corrupt-then-reload proves real re-quantization; layerwise
  reload leg mirrors `_weight_update_lifecycle` in
  `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`.
- **D — logprob fidelity**: `avg_prob_mult_error = mean(exp(|Δlogprob|))` of
  per-token and static-scale legs vs BF16 reference over the same sampled
  token sequences. Gates: per-token ≤ 1.20 AND ≤ static × 1.05.

Run: `sbatch sbatch_nvfp4_pertoken.sh` (or `RUN_CHECKS=A,B ./sbatch_...` to select).

## Findings

(TODO — filled as checks complete)

| Check | Status | Notes |
|---|---|---|
| Stage-0 gates | PASS (2026-07-18, job 2404072) | vLLM 0.23.1rc1.dev1261+gc71a583aa; `Nvfp4OnlineMoEMethod` importable; `make_nvfp4_moe_kernel(per_token_activation=...)` present; `has_flashinfer_trtllm_fused_moe()=True` on GB200 SM100 |
| A smoke (granite + Qwen3-30B) | PASS (job 2404103) | `nvfp4_per_token` loads and generates sanely on both; `Nvfp4OnlineMoEMethod` active on every MoE layer; `FLASHINFER_TRTLLM` backend selected |
| A TP=2 | PASS — **TP works now** (job 2404103) | Survey's "TP raises NotImplementedError" is outdated; nightly initializes TP=2 and generates sanely. Removes the biggest wiring blocker for multi-GPU rollout |
| B.1 external-quant layout + determinism | PASS (job 2404103) | `_quantize_moe_weight_to_nvfp4` output = ModelOpt ckpt layout (uint8 packed / fp8-e4m3 block / fp32 global); bitwise deterministic |
| B.2 hybrid per-token overlay (GO/NO-GO) | **PASS** (job 2404103) | Pre-quantized ModelOpt NVFP4 weights + per-token dynamic activations generate sanely through FlashInfer TRT-LLM; `ModelOptNvFp4PerTokenFusedMoE` active. The NeMo-RL to-be flow is kernel-feasible |
| C reload determinism | FINDING (job 2404103, retest pending) | `reload_weights` fails on stride-0 expanded scale views: layerwise-reload finalize `param.data.copy_()` cannot write into them ("more than one element ... single memory location"). Affects upstream `nvfp4_per_token` (upstream bug, matches RFC #48312) AND the v1 overlay — overlay fixed with `.contiguous()` (same as #2983's approach); upstream path also blocked by `_already_called_process_weights_after_loading` guard skipping re-quantization on reload |
| D fidelity (per-token / static / hybrid vs BF16) | pending (job 2404119) | |

### Log markers for Stage-2 assert_grep

(TODO — recorded from check A DEBUG run)

### Measured thresholds for Stage-2

(TODO — from check D)
