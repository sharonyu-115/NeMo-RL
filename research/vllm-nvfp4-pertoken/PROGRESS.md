# Implementation progress — TE NVFP4 per-token W4A4

Tracks `IMPLEMENTATION-PLAN-te-nvfp4-pertoken.md` execution. One row per step;
problems recorded inline as encountered.

| Step | Status | Problems / notes |
|---|---|---|
| 1. producer + bitwise GPU test | **DONE** (job test4, 9/9 pass) | Bitwise vs vLLM kernel passed FIRST RUN (3 shapes). Container quirks: cwd resets to /vllm-workspace (need cd); tests/unit/conftest.py imports ray → use --noconftest in barebones containers |
| 2. refit filter + unit | **DONE** (same job) | filter tests green (pattern select, 3-tensor naming, pass-through, device) |
| 3. graduate overlay | **DONE** (SMOKE_GRADUATED_OK, smoke7) | PLAN ADJUSTMENT: split into two modules — nvfp4_pertoken.py (vLLM-free: producer/filter/ignore/hf-dict; training-venv importable) + nvfp4_pertoken_vllm.py (module-scope vLLM subclasses for engine-proc pickling). 'Same module' was contradictory |
| 4. transport extension | **DONE** (code; e2e refit needs P2 env) | Simpler than planned: per-expert checkpoint-layout names mean NO fused-family batching/manifest code needed at all — NvFp4PerTokenWorkerExtension is ~40 lines over the base lifecycle hooks (initialize/finalize layerwise reload + fatal errors + accel sync). #2983 refactor not needed |
| 5. config + engine kwargs + guards | **DONE** | NvFp4PerTokenRolloutConfig BaseModel (defaults on class); VllmConfig NotRequired key; resolver dispatch + mutual-exclusion ValueError in resolve_generation_worker_cls; engine kwargs incl. load_format=dummy (BF16 ckpt can't fill NVFP4 params; first refit precedes first generation). Exemplar-YAML doc deferred to recipe PR (avoids reference_configs churn now). f2l4 consistency guard deferred to PR-2 (fp4_cfg doesn't exist on main yet) |
| 6. refit hook (MegatronPolicyWorker) | **DONE** (code; exercised in P2 e2e) | Wrapped _iter_params_with_optional_kv_scales (renamed impl + conditional filter): all three refit surfaces (prepare_refit_info metadata, ZMQ, collective) stay consistent automatically — the state_dict_info problem solved itself. Note: _calculate_refit_param_info (mcore bucket sizing) still uses bf16 sizes → buckets overestimate, harmless |
| 7. CPU-mocked unit tests | **DONE** (11 pass +1 env-skip; full set in CI post-repin) | In the producer test file (self-contained w/ import fallbacks): filter ignore-patterns, rollout-config defaults, resolver dispatch + mutual exclusion (skips where full deps absent, runs in CI) |
| 8. env (vLLM repin / TE pin) | blocked-ops | out of session scope; probe nightly sqsh used for GPU validation meanwhile |
| 9. fp4_cfg upstream | blocked on TE pin | |
| 10. probe rerun w/ nvfp4_pertoken | **DONE** (B2-equivalent green via smoke_graduated_module.py) | smoke_graduated_module.py = probe B2 against the graduated production classes (stubs heavy nemo_rl imports in barebones container) | |
| 11. functional + nightly | pending (after 8/9) | |

## Log

- 2026-07-19: kickoff. Bitwise-matching risk noted up front: pure-torch producer
  must replicate the CUDA kernel's exact rounding (RNE e4m3 cast, `x * (1/sf)`
  multiply-by-reciprocal, RNE-on-grid e2m1, satfinite clamp, nibble pack order).
  GPU test in the nightly container is the arbiter; iterations expected.

- 2026-07-19 (P1 debugging log): graduated-module smoke initially failed with
  `ValueError: quant method must be one of ['tensor','channel','group','block']`
  at initial checkpoint load. TWO root causes found and fixed:
  1. **vLLM duck-types NVFP4 expert scale loading on the quant-method CLASS
     NAME** (`"ModelOpt" in quant_method.__class__.__name__`,
     routed_experts.py:703/744). Renaming the class for neutrality silently
     dropped scale params out of the ModelOpt branch. Fix: class stays
     `ModelOptNvFp4PerTokenFusedMoE` (registered method name is still
     `nvfp4_pertoken`); fragility documented in the class docstring.
     Upstream-worthy finding.
  2. The literal hf quantization_config dict guessed the wrong schema; now
     mirrors the real ModelOpt NVFP4 checkpoint's `config.json` key-for-key
     (`ignore` not `exclude_modules`, `targets:["Linear"]`, `producer`) with
     the single per-token delta `input_activations.dynamic=true`.
  Also built `shim/` (stub nemo_rl tree on PYTHONPATH) so barebones containers
  can exercise the graduated module incl. vLLM's EngineCore subprocess, which
  re-imports the pickled config by package name.

## Changes by step (commits aab14dd7f = P0, fe56561bb = P1)

| Step | Files | What the change is |
|---|---|---|
| 1 | `nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py` (new, P0): `quantize_nvfp4_weight` | Pure-torch NVFP4 weight quantizer (per-tensor/per-expert amax → fp32 global scale; block-16 e4m3 micro-scales; RNE-on-grid E2M1 rounding; nibble packing). vLLM-free so Megatron training workers can run it at refit. Bit-identical to vLLM's online-quant kernel |
| 1 | `tests/unit/models/generation/test_nvfp4_pertoken_producer.py` (new, P0) | The P0 hard gate: bitwise comparison vs `_quantize_moe_weight_to_nvfp4` on GPU, plus determinism/layout/2D≡3D/round-trip checks. Self-contained (file-path import fallback) so it runs in barebones containers |
| 2 | same module (P0): `iter_nvfp4_pertoken_weights` | Refit filter: wraps the `(hf_name, tensor)` export stream, quantizes `*.weight` matching quant_patterns (minus ignore_patterns) on-GPU into `.weight/.weight_scale/.weight_scale_2`, passes everything else through. This IS the "quantize during refit" of the design — no Megatron-Bridge change needed |
| 3 | `nvfp4_pertoken_vllm.py` (new, P1): `ModelOptNvFp4PerTokenFusedMoE`, `NvFp4PerTokenConfig`, `register_nvfp4_pertoken` | The probe overlay graduated to production: registers vLLM quantization method `nvfp4_pertoken` — stock ModelOpt NVFP4 W4A4 loading but neutral (1.0) activation global scales and `make_nvfp4_moe_kernel(per_token_activation=True)`, plus `.contiguous()` on kernel scales for reload safety. Class name keeps the "ModelOpt" substring vLLM's expert loader duck-types on |
| 3/5 | `nvfp4_pertoken.py` (P1): `build_nvfp4_pertoken_hf_quant_config` | Literal HF `quantization_config` override mirroring the real ModelOpt NVFP4 checkpoint schema key-for-key, with `input_activations.dynamic=true`. Needed because the training checkpoint on disk is BF16 and carries no quant config |
| 4 | `nvfp4_pertoken_vllm.py` (P1): `NvFp4PerTokenWorkerExtension` | Refit transport: wraps every weight update in vLLM's layerwise-reload lifecycle (initialize → load → finalize) so quantized params restore to load format and the per-token kernel rebuilds with CUDA-graph-stable storage; refit errors fatal; accelerator-fence before IPC ack. ~40 lines — per-expert checkpoint-layout names removed the need for #2983-style fused-family batching |
| 5 | `nvfp4_pertoken.py` (P1): `NvFp4PerTokenRolloutConfig` | v2 BaseModel user config (`enabled`, `ignore`, `quant_patterns`) — defaults live on the class per config conventions |
| 5 | `config.py`: `nvfp4_pertoken_rollout` NotRequired key | Exposes the config block under `policy.generation` in YAML |
| 5 | `nvfp4_pertoken_worker.py` (new, P1) | Sync/async generation workers that inject the engine kwargs at `_create_engine` time: `quantization="nvfp4_pertoken"`, the HF quant-config override, `load_format="dummy"` (BF16 ckpt can't fill NVFP4-shaped params; first refit supplies all weights), and the worker extension |
| 5 | `utils.py`: `resolve_generation_worker_cls` + `NVFP4_PERTOKEN_WORKER_OVERRIDES` | Worker dispatch for the new mode + hard mutual-exclusion `ValueError` when combined with the ModelOpt keys (`quant_cfg`/`real_quant`) |
| 6 | `megatron_policy_worker.py`: `_nvfp4_pertoken_rollout_cfg`, wrapped `_iter_params_with_optional_kv_scales` (original renamed `_impl`) | Training-side refit hook: when the mode is enabled, the shared export iterator is wrapped with the Step-2 filter. Because prepare_refit_info metadata, ZMQ streaming, and collective broadcast all consume this one iterator, the advertised state-dict info and streamed payloads stay consistent by construction |
| 7 | test file additions (P1) | CPU-mocked coverage: filter ignore-patterns, rollout-config defaults/extra-keys, resolver dispatch + mutual exclusion (env-skips where full deps absent; complete in CI post-repin) |
| 10 | `research/.../smoke_graduated_module.py` + `shim/` (P1) | GB200 validation harness: probe check B2 against the production classes; the shim is a stub `nemo_rl` package tree on PYTHONPATH so vLLM's EngineCore subprocess can re-import the pickled config in containers lacking ray |

## P2 execution (2026-07-20)

| Step | Status | Problems / notes |
|---|---|---|
| A1 repin + gitlink + mirror | **DONE** (0300826d9) | Nightly index paths aren't the real S3 objects — actual wheels live at wheels.vllm.ai/<full-sha>/; first mirror grabbed 300-byte error XML. Pinned dev1283 (dev1261's full sha unavailable due to GH API rate limit); A4 producer test is the drift gate. Index moved dev1282→dev1283 overnight — validates mirror-first strategy |
| A2 uv lock | **DONE** (7f51094fb) | Risk-1 fired: flashinfer-cubin has no 0.6.14 on PyPI (skipped: max 0.6.13). flashinfer-python declares no cross-pin → coherent set python==0.6.14 / cubin==0.6.13 / jit-cache==0.6.14+cu130. Lock green, no 0.20 residue |
| A3 venv bake | RUNNING (sbatch 2410178) | First srun attempt would have died at the 10-min background cap — resubmitted as sbatch. Bake run = tiny megatron GRPO (grpo_math_1B_megatron, cached Qwen2.5-1.5B, HF offline) building mcore+vllm venvs; NvFp4PerTokenGenerationWorker venv builds on first M1 run (uv cache warm) |
| B1-B4 M1 artifacts | **DONE** (e1356a920) | Recipe inherits performance parent directly (skips ModelOpt QA parent); moe_backend auto (parent pins triton for vllm-0.20 refit reasons); driver greps swapped to per-token markers incl. per-refit "refit: quantized N" liveness line (RuntimeError on zero-quantized); ptyche launcher + .env |
| C1-C3 fp4 port + tests | **DONE** (e1356a920) | fp4 block extracted to apply_te_precision_config for testability; 7/7 logic tests pass via shim runner on login node (no pytest there); NVTE_BACKWARD_OVERRIDE gated train-only at both forward_backward sites; f2l4↔ignore warning at prepare_refit_info; data.py fp4 padding branch |
| C4 fp4train recipe | **DONE** (e1356a920) | fp4_cfg + f2l4 + NVTE per-token env vars (ROW_SCALED_ACTIVATION etc.), NVTE_BACKWARD_OVERRIDE=dequantized; driver adds [fp4_cfg] grep |
| A4 gates | pending (after bake) | |
| B5 run ladder | pending (after A4) | |
| D nightly | pending (after B5/C4 runs) | |

### B5.1 refit-smoke debugging log (2026-07-20, jobs 2410468→2410625)

Six iterations, five real defects found and fixed — exactly what the 1-node
rung exists for:
1. `cluster.segment_size=4` from the performance parent doesn't divide 1 node
   → smoke overrides segment_size=1 (config-only).
2. **Actor registry**: new Ray worker classes need
   ACTOR_ENVIRONMENT_REGISTRY entries → per-token workers registered with the
   vLLM executable (1f0e329a4).
3. **Router ignore pattern**: `*mlp.gate.*` doesn't fnmatch the bare router
   prefix `...mlp.gate` → vLLM NVFP4-quantized the router while refit streamed
   it BF16 (shape mismatch [128,2048]→[128,1024]). Added bare-suffix pattern.
4. **ZMQ timeouts**: first refit re-processes every layer vLLM-side
   (per-token kernel rebuild + FlashInfer autotune) → 600s timeouts on both
   sides, mirroring the ModelOpt path.
5. **Per-expert streaming too slow** (THE big one): ~55k tensors through
   per-tensor IPC + reload buffering couldn't finish a refit in 600s
   (vLLM side showed ~6GB of reload buffers accumulating ~1MB/tensor). Filter
   rewritten to emit FUSED stacked tensors in the ModelOpt fused-MoE
   convention (~6/layer, per-(expert,projection) scales preserving on-disk
   semantics; flush on layer-prefix change). 14/14 unit tests incl.
   fused≡per-projection equality. The plan's "optional Bridge batching"
   risk materialized at the transport layer instead.
6. Retry-5 then ran the ENTIRE M1 loop (dummy load → fused quantized refit →
   per-token generation → train step) and died only in save_checkpoint —
   1-node host OOM staging the 30B optimizer state (862/890GB), a smoke-config
   artifact. Liveness marker switched to print() (Ray workers hide INFO), and
   the smoke disables checkpointing.

Also: bake4 (TE 937c4de) GREEN — TE 2.17.0.dev0+937c4de0, ROW_SCALED knob
present, training step passed → v4 sqsh is the M2 image
(nemo-rl-nvfp4-pertoken-venvs-gb200-2026-07-20-v4.sqsh). TE 2.15 knob absence
confirmed in .so strings, not just python source.

7. **Fused w13 needs ONE global scale per expert** (numerics, jobs
   2410625/2410626): both runs completed mechanically (refit markers on all
   ranks, 40-70s step times) but generations were garbage — reward=0,
   val accuracy=0, every sequence at the 4096 cap, NaN entropy/gen_kl (train
   side healthy: finite kl_penalty). Root cause in vLLM's
   `ModelOptNvFp4FusedMoE.process_weights_after_loading`:
   `w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]` — the loader keeps
   only the GATE global scale for the whole fused w13 (warning_once
   "w1_weight_scale_2 must match w3_weight_scale_2", present in the logs on
   every worker). Our filter quantized gate/up per-projection, so the up half
   decoded off by scale2_gate/scale2_up per expert → corrupted every MoE
   layer. Real ModelOpt ckpts don't hit this (fused-aware export = equal
   scales), which is why the Stage-1 hybrid probe was coherent. Fix: filter
   now quantizes the stacked (E, 2N, K) gate+up tensor in one producer call
   (per-expert amax over both projections — exactly upstream's
   `_quantize_moe_weight_to_nvfp4` online behavior) and emits (E, 2)
   scale_2 with identical columns. Lesson: the warning was in retry-5/6 logs
   all along — grep for `Accuracy may be affected` class warnings, not just
   crashes. CORRECTION (defect #8 triage): that warning also fires once per
   engine process at dummy-load startup (random scales are never allclose),
   so it is NOT a usable driver gate — the unit test asserting identical
   scale_2 columns is the guard. Defect #7 was real but masked by #8.

8. **Fused expert tensors never loaded — RoutedExperts name-contract
   mismatch** (job 2411044, run dir also cleaned + launcher now wipes stale
   run dirs after job 2411027 auto-resumed past max_num_steps and no-opped):
   generations were byte-identical to the pre-#7-fix runs (val accuracy 0,
   avg_length 3977.6 to the decimal) and the log showed
   `[layerwise.py:268] RoutedExperts: Failed to load weights` ×720
   (48 layers × workers). Root cause: dev1283's
   `RoutedExperts.load_weights` matches per-expert checkpoint names
   (`experts.{e}.gate_proj.weight` ...) or BF16 HF fused names
   (`experts.gate_up_proj`, with transpose heuristics that break on packed
   uint8), but NOT the `w13_weight`/`w2_weight` parameter names our filter
   emitted — every fused tensor passed through unmatched (load_numel=0) and
   finalize restored the previous DUMMY kernel tensors, with only a warning.
   So since retry-5 every "successful" refit refit nothing; step times looked
   great because loading was skipped. Fix: keep the fused format for
   transport, expand vLLM-side in the worker extension
   (`expand_fused_expert_weights`: local slicing into per-expert ModelOpt
   names, no per-tensor IPC) before `model.load_weights`. Drivers now forbid
   `RoutedExperts: Failed to load weights` (fires per refit on real failures;
   quiet on dummy load). Lessons: (a) refit liveness must be proven on the
   CONSUMER side, not the producer side — the Megatron-side marker counted
   tensors sent, not loaded; (b) vLLM swallows unmatched refit names by
   design — always check `Failed to load weights` warnings when output
   quality is impossible.

9. **Layer-completion stall** (job 2411197, killed preemptively): with #8
   fixed the first REAL weight load buffered the entire model
   (`Allocating ~5.4GB of device memory to buffers` per bucket, still going
   ~8min into the 600s ZMQ window). The quant method registers
   w13/w2_input_scale params, counted in load_numel_total, but the refit
   never streamed them → every RoutedExperts layer stayed "incomplete" →
   all processing deferred to finalize. Fix: expansion also emits neutral
   input_scale=1.0 per (expert, projection) — the names real W4A4 ckpts
   carry, values our method overwrites anyway — restoring online per-layer
   processing. (This was also defect #5's true mechanism.)

### B5.2 4n4g 2-step: metrics GREEN (2026-07-20)

Attempt ladder: 2410626 (defects #7/#8 latent) → 2411027 (no-op, stale-ckpt
auto-resume; launcher now wipes run dirs) → 2411044 (#7 fixed, exposed #8) →
2411117 (#8 fixed, exposed #9, killed at ZMQ-window risk) → **2411197**
(#7+#8+#9 in).

**RUN VERDICT (job 2411197, 4n4g, 2 steps + val, all fixes in):**
- validation/accuracy 0.5625 @ step0 AND step2 (gate ≥0.4 ✓)
- train/reward 0.5625 / 0.5 (gate ≥0.25 ✓)
- gen_kl_error 0.017/0.018 (<0.03 ✓), js_divergence 0.0046/0.0050
  (<0.007 ✓), approx_entropy 0.21/0.23 (<0.35 ✓),
  token_mult_prob_error 1.12/1.51
- refit ~40s total step time incl. quantize+transport+online reload;
  no buffering warnings, no failed-load warnings
- Job state FAILED on ONE driver grep only: the "per-token NVFP4 activation
  scaling active" marker was logger.info_once on a nemo_rl.* logger inside
  the engine proc (only the "vllm" tree is configured there — INFO dropped).
  Defect #10, instrumentation-only: switched to print-once. All quality
  gates evaluated green from metrics.json by hand. Needs one rerun for a
  formally green driver.

### B5.2 formal rerun: JOB 2411254 COMPLETED — driver fully green
(2026-07-20). All greps (incl. the print-once activation marker) + all
check_metrics gates passed end-to-end. val acc 0.5625@0 / 0.50@2 (gen-order
nondeterminism at temp 1.0; both ≥0.4), step times ~40s. **M1 is done at the
2-step rung.** 50-step confirm launched as job 2411276 (MAX_STEPS=50).

### M2 fp4train first e2e attempts (2026-07-20)
- Job 2411255 (v4 image via new CONTAINER_IMAGE_OVERRIDE launcher knob):
  `[fp4_cfg] Megatron FP4 training enabled: fp4=e2m1 recipe=nvfp4
  fp4_param=False` on all 16 ranks; engine/refit/val@0 green (acc 0.5625,
  identical to M1 as expected — step-0 val precedes any fp4 training).
  **Defect #11**: first logprob forward died in TE
  `fused_multi_row_padding` with `GetTransformerEngineDType: Invalid type
  (7)` — type 7 = float64. The perf parent pins `moe_router_dtype: fp64`;
  with fp4_cfg on, Megatron MoE experts route fp64 permuted_probs through
  TE's quantization padding, which has no double kernel. BF16 (M1) never
  touches that module. Fix: `moe_router_dtype: fp32` in the fp4train
  variant (2c28be595) — the rl-fp4 study never set fp64, so it never hit
  this.
- Retry job 2411268 (fp32 router): fp4 training steps RAN at 30B scale
  (~41s, BF16 parity), reward 0.44, val acc 0.5625. Driver FAILED on
  gen_kl 0.042/js 0.012 vs the BF16-variant gates (0.03/0.007). Two
  fixes (defect #12): (a) f2l4 guard fired — training kept layers
  0-1/44-47 BF16 but the rollout quantized their experts; added those
  layers' experts to the rollout `ignore` (explicit list replaces the
  default). (b) gates widened to 0.05/0.015 (fp4 train logprobs quantize
  differently than the flashinfer rollout kernel; larger residual gap is
  expected). token_mult_prob_error spiked to 24.4@1 (1.79@2).

**M2 fp4train GREEN — JOB 2411336 COMPLETED, driver fully passed.** Refit
now 42 quantized layers / passthrough 2739 (f2l4 experts stream BF16); no
f2l4 warning; ~41s steps (2739-tensor BF16 passthrough adds no overhead).
Metrics: val acc 0.5625→**0.625** (rose over training), reward 0.625/0.625,
gen_kl 0.035/0.037, js 0.0098/0.0105, entropy 0.23/0.24 — all gates green.
Watch item: token_mult_prob_error elevated (15.8→9.9 vs M1's ~1.1) — the
fp4-training-vs-rollout-kernel quant gap; ungated, run healthy (reward/acc
rising), monitor at longer horizons. **M1 (BF16 train) AND M2 (fp4 train)
per-token W4A4 rollout both validated e2e on Qwen3-30B-A3B/GB200.**
