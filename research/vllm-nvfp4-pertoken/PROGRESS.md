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
