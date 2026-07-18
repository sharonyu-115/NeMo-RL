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
| Container | `/lustre/fsw/general_sa/shuangy/images/vllm-nightly-2026-07-18.sqsh` |
| Image source | `docker://vllm/vllm-openai:nightly` (digest: TODO after import) |
| vLLM commit | TODO (must postdate 2026-07-16, PR #48538) |
| Hardware | 1× GB200 node (feature is SM100-only, MoE-only, TP=1 only) |
| Models | `Qwen/Qwen3-30B-A3B` (BF16), `nvidia/Qwen3-30B-A3B-NVFP4` (static-scale baseline) |

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
| Stage-0 gates | | |
| A smoke | | |
| B.1 external-quant equivalence | | |
| B.2 hybrid per-token overlay | | |
| C reload determinism | | |
| D fidelity (per-token / static vs BF16) | | |

### Log markers for Stage-2 assert_grep

(TODO — recorded from check A DEBUG run)

### Measured thresholds for Stage-2

(TODO — from check D)
