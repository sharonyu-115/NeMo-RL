# Implementation progress — TE NVFP4 per-token W4A4

Tracks `IMPLEMENTATION-PLAN-te-nvfp4-pertoken.md` execution. One row per step;
problems recorded inline as encountered.

| Step | Status | Problems / notes |
|---|---|---|
| 1. producer + bitwise GPU test | **DONE** (job test4, 9/9 pass) | Bitwise vs vLLM kernel passed FIRST RUN (3 shapes). Container quirks: cwd resets to /vllm-workspace (need cd); tests/unit/conftest.py imports ray → use --noconftest in barebones containers |
| 2. refit filter + unit | **DONE** (same job) | filter tests green (pattern select, 3-tensor naming, pass-through, device) |
| 3. graduate overlay | IN PROGRESS | PLAN ADJUSTMENT: split into two modules — nvfp4_pertoken.py (vLLM-free: producer/filter/ignore/hf-dict; training-venv importable) + nvfp4_pertoken_vllm.py (module-scope vLLM subclasses for engine-proc pickling). 'Same module' was contradictory |
| 4. transport extension | pending | plan fallback taken: standalone helpers, #2983 refactor deferred (PR unmerged) |
| 5. config + engine kwargs + guards | pending | |
| 6. refit hook (MegatronPolicyWorker) | pending | |
| 7. CPU-mocked unit tests | pending | |
| 8. env (vLLM repin / TE pin) | blocked-ops | out of session scope; probe nightly sqsh used for GPU validation meanwhile |
| 9. fp4_cfg upstream | blocked on TE pin | |
| 10. probe rerun w/ nvfp4_pertoken | pending (after 3) | |
| 11. functional + nightly | pending (after 8/9) | |

## Log

- 2026-07-19: kickoff. Bitwise-matching risk noted up front: pure-torch producer
  must replicate the CUDA kernel's exact rounding (RNE e4m3 cast, `x * (1/sf)`
  multiply-by-reciprocal, RNE-on-grid e2m1, satfinite clamp, nibble pack order).
  GPU test in the nightly container is the arbiter; iterations expected.
