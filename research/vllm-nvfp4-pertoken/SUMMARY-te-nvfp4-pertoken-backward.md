# Summary: NVFP4 per-token BACKWARD campaign

Branch `shuangy/te-nvfp4-pertoken-backward` (worktree `nemo-rl-te-bwd`), based on
`1d3f0eea8`. Goal: enable **real FP4 per-token dgrad/wgrad** in the Megatron
MoE-expert GEMMs via TransformerEngine PR #3045 (fork `cael-ling@690ffea`), exposed
as a typed `Fp4Config` field, and run the fwd/bwd + router-replay A/B on the 8n4g
DAPO-512/20k Qwen3-30B-A3B campaign. Companion docs:
`DESIGN-te-nvfp4-pertoken-backward.md` (mechanism), `ROW-SCALED-vs-FULL-PER-TOKEN.md`
(numerics).

## Commit series (base `1d3f0eea8` → HEAD)

| Commit | What |
|---|---|
| `475077431` build(te) | pin TE → `cael-ling@690ffea` (PR #3045) in pyproject |
| `80fed9c53` feat(megatron) | typed `fp4_cfg.backward` wiring (fp4_env.py + lm_policy + worker) |
| `2cba49557` docs(research) | DESIGN + ROW-SCALED-vs-FULL-PER-TOKEN |
| `afcccaf1e` build(te) | TE 690ffea wheel prebuild + probe bake scripts |
| `eb0fd9fe3` fix(megatron) | grouped-expert empty `_extra_state` checkpoint patch |
| `4983cc781` chore(recipe) | fp4bwd long run → 1500 steps |
| `521d57d6e` feat(recipe) | fp4bwd+r3 and per-token-fwd (leg 2) A/B recipes |

## Code changes

- **`nemo_rl/models/policy/__init__.py`** — `Fp4Config` gains `backward`
  (`dequantized`|`high_precision`|`nvfp4_pertoken`) + `per_token_{rht,sr,weight_2d}`
  (all NotRequired; absent ⇒ no NVTE_* emitted ⇒ byte-for-byte with today).
- **`nemo_rl/models/megatron/fp4_env.py`** (new, stdlib-only so the driver can
  import it) — pure `fp4_cfg_to_env_overrides`, `apply_fp4_backward_env_overrides`
  (raw env_vars win for SETs; the per-token UNSET clears an inherited
  `NVTE_BACKWARD_OVERRIDE`), and `assert_te_supports_fp4_backward` gate.
- **`nemo_rl/models/policy/lm_policy.py`** — inject derived NVTE_* into the worker
  runtime `env_vars` on the driver (same Ray runtime-env channel as the fwd knobs;
  lands in os.environ at process start).
- **`nemo_rl/models/policy/workers/megatron_policy_worker.py`** — capability gate;
  make the existing train-only `NVTE_BACKWARD_OVERRIDE` capture per-token-aware
  (force no override so FP4 backward engages); install the checkpoint patch.
- **`nemo_rl/models/megatron/te_grouped_ckpt_patch.py`** (new) — checkpoint fix
  (below).
- **`tests/unit/models/megatron/test_fp4_env.py`** (new) — translation/gate tests.

### Enable switch (env-var-only, design D1 — no Megatron/recipe-class change)

`backward: nvfp4_pertoken` → `NVTE_NVFP4_PER_TOKEN=1` + **unset**
`NVTE_BACKWARD_OVERRIDE`. Empirically confirmed: a plain `NVFP4BlockScaling` (what
Megatron builds) flips `backward_override 'dequantized'→None` under the switch.
Why the driver channel and not `setup.py`: `apply_te_precision_config` runs *after*
the worker's train-only `NVTE_BACKWARD_OVERRIDE` capture, so writing os.environ
there would defeat it.

## Fixes / blockers overcome (with reasons)

1. **TE @690ffea wouldn't build** (multi-hour "timeouts"):
   - **NCCL EP** submodule (`nccl_ep.cc`) needs newer NCCL GIN/device-comm symbols
     than the container ships → compile error. Fix: `NVTE_WITH_NCCL_EP=0` (EP is
     comm-overlap, irrelevant to NVFP4 GEMMs; arch 10.0 ≥ 90 so no auto-skip).
   - **`nproc=1`**: `srun --exclusive` without `-c` gives the task a 1-CPU cgroup →
     every from-source compile single-threaded. Fix: `--cpus-per-task=128`.
   - Fastest path: build the TE wheel once into the persistent lustre uv cache
     (`prebuild_te690_wheel.sh`), then the bake reuses it. TE built in 68 min.
2. **Grouped-expert checkpoint save crash** (`fix eb0fd9fe3`): per-token FP4 leaves
   the grouped MoE-expert `_extra_state` empty (scales computed on the fly — no
   persistent amax/scale state, unlike delayed scaling). Megatron's
   `_split_extra_state` sees the fp8 flag, decodes the empty state to `None`, then
   subscripts it → `'NoneType' object is not subscriptable`. Row-scaled forward
   populates `_extra_state` so it only bit per-token. Patch: return
   `[state] * num_gemms` when the decoded state is None (mirrors the no-fp8
   fallback; load-side `merge_extra_states` already guards None). Installed on the
   worker when fp4 is enabled.
3. **Recipe subtleties**:
   - Nested `_override_` does NOT work (merge_with_override honors it only for
     top-level sections) → env_vars deep-merge; the code clears the stale
     `NVTE_BACKWARD_OVERRIDE`.
   - `max_num_steps` baked in the recipe (not per-submission) so every resume reads
     the same value → no Megatron `wd_incr_steps` resume assert (constant-LR sched).

## Validation

- **Probe image** `nemo-rl-te690ffea-probe.sqsh` (TE `2.18.0.dev0+690ffea4`);
  PR #3045 recipe classes import; D1 confirmed.
- **P3 smoke** (job 2444399): 1 step, gen_kl 0.0102, FP4 backward engaged.
- **Save+resume smoke** (jobs 2445599/2445600): saved step_2..16 (no crash),
  resumed step_16 → "Step 17/1500".
- **Long run** (job 2444465 first attempt died at the checkpoint bug → fixed →
  2445764 saved step_10 cleanly).

## Environment / build

- Probe image built via `env_refresh_te690.sh` on the te690 uv-cache wheel;
  requires `NVTE_WITH_NCCL_EP=0` + `srun --cpus-per-task=128`.
- Launch: `research/vllm-nvfp4-pertoken/run_dapo_longrun.sh` gained
  `PRECISION=nvfp4_bwd` (te690-image guard). Other legs use
  `PRECISION=nvfp4_bwd RECIPE_OVERRIDE=<recipe>`.

## A/B experiment matrix (8n4g DAPO-512/20k, 1500 steps each)

| Leg | Forward | Backward | Router replay | Recipe / run |
|---|---|---|---|---|
| 1 | row-scaled | dequant (BF16) | off | `-nvfp4-pertoken` (done) |
| 2 | full per-token | dequant (BF16) | off | `-nvfp4-pertoken-fp4fwd` (2445985) |
| 3 | full per-token | **FP4** | off | `-nvfp4-pertoken-fp4bwd` (2445764) |
| 3+r3 | full per-token | **FP4** | **on** | `-nvfp4-pertoken-fp4bwd-r3` (2445971) |

Isolates: forward-cast swap (1→2), FP4 backward (2→3), router replay (3→3+r3).
Judge on `token_mult_prob_error` / `gen_kl` / `grad_norm` / reward convergence.

## Watch-outs

- `690ffea` is an UNMERGED fork HEAD (probe-grade); re-pin if it rebases.
- Do NOT branch-switch the original checkout (`/lustre/.../nemo-rl`) — its running
  chain reads `patches.py` live.
- Related memories: `te690-nccl-ep-build-blocker`,
  `nvfp4-pertoken-grouped-expert-ckpt-bug`, `megatron-max-steps-resume-assert`.
