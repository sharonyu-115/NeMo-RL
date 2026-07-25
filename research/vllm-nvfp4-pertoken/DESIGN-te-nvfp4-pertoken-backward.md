# Design: NVFP4 per-token BACKWARD (dgrad/wgrad) in NeMo-RL Megatron

Branch: `shuangy/te-nvfp4-pertoken-backward` (worktree `nemo-rl-te-bwd`).
Extends the per-token FORWARD work (`DESIGN-te-nvfp4-pertoken.md`,
[`CONFIG-fp4train.md`]) to enable **real FP4 per-token dgrad/wgrad** via
TransformerEngine fork `cael-ling@690ffea4` (PR #3045, TE `2.18.0.dev0` line).

Numerics background — why "backward" also swaps the forward cast — is in
[`ROW-SCALED-vs-FULL-PER-TOKEN.md`].

## Goal & scope

Let a Megatron GRPO recipe turn on NVFP4 per-token backward for the MoE-expert
GEMMs through a **typed config field**, defaulting to today's behavior
byte-for-byte. Primary metric is numerics / convergence / train–gen alignment,
not throughput. Non-goals: no Megatron/recipe-class change; no new TE build
features (RHT/SR/weight-2D per-token are opt-in ablations, wired but off).

## Mechanism (design D1): env-var translation, no Megatron change

TE @690ffea makes per-token selectable purely from env vars — a plain
`NVFP4BlockScaling` (what Megatron builds with `--fp8-recipe nvfp4`) flips into
per-token mode when `NVTE_NVFP4_PER_TOKEN=1`, and FP4 backward runs when
`NVTE_BACKWARD_OVERRIDE` is unset (`backward_override is None`). So NeMo-RL only
needs to set the right env vars in the Megatron worker; Megatron keeps building
the same recipe object. Empirically confirmed on the probe image (constructing
`NVFP4BlockScaling()` under the switch flips `backward_override 'dequantized'→None`,
`disable_2d/rht/sr→True`, `row_scaled_activation→False`).

Enable switch: **`NVTE_NVFP4_PER_TOKEN=1` + UNSET `NVTE_BACKWARD_OVERRIDE`**
(+ optional `NVTE_NVFP4_PER_TOKEN_{RHT,SR,WEIGHT_2D}=1`).

## Config surface (design D2): typed `Fp4Config` fields

`nemo_rl/models/policy/__init__.py` — `Fp4Config` gains (all `NotRequired`):

| field | values | meaning |
|---|---|---|
| `backward` | `dequantized` (=today) / `high_precision` / `nvfp4_pertoken` | backward precision mode |
| `per_token_rht` | bool | opt in RHT on the per-token backward |
| `per_token_sr` | bool | opt in stochastic rounding |
| `per_token_weight_2d` | bool | opt in 2D weight cast |

**Field absent ⇒ nothing emitted** — existing fp4 recipes that rely on TE's own
backward default are byte-for-byte unchanged.

## Translation & precedence

New dependency-light module `nemo_rl/models/megatron/fp4_env.py` (stdlib only;
TE imported lazily) — importable on the driver (which lacks the megatron venv, so
`setup.py`/`megatron.bridge` can't be imported there):

- `fp4_cfg_to_env_overrides(fp4_cfg) -> (to_set, to_unset)` — pure mapping:
  - `backward="dequantized"` → set `NVTE_BACKWARD_OVERRIDE=dequantized`
  - `backward="high_precision"` → set `NVTE_BACKWARD_OVERRIDE=high_precision`
  - `backward="nvfp4_pertoken"` → set `NVTE_NVFP4_PER_TOKEN=1`, **unset** `NVTE_BACKWARD_OVERRIDE`
  - `per_token_{rht,sr,weight_2d}=True` → set `NVTE_NVFP4_PER_TOKEN_{RHT,SR,WEIGHT_2D}=1`
- `apply_fp4_backward_env_overrides(env_vars, fp4_cfg)` — merges into the worker
  runtime `env_vars`. **SET** vars use `setdefault` (raw recipe `env_vars` win,
  design D3); the per-token **UNSET** is authoritative — it **pops** any
  pinned/inherited `NVTE_BACKWARD_OVERRIDE` (with a warning), because otherwise
  the field would be a silent no-op and YAML inheritance can't drop a nested key.
- `assert_te_supports_fp4_backward(fp4_cfg)` — capability gate: raises if
  per-token backward is requested but the installed TE lacks PR #3045
  (`NVFP4PerTokenBlockScaling` not importable).
- `fp4_cfg_wants_per_token_backward(fp4_cfg)` — predicate used worker-side.

### Where each piece runs (and why)

- **Driver** — `lm_policy.py` (Policy `__init__`, Megatron branch): calls
  `apply_fp4_backward_env_overrides` on the runtime `env_vars` dict right where
  `megatron_cfg.env_vars` is assembled. This is the **same Ray runtime-env
  channel** today's forward knobs ride, so the derived vars are in the worker's
  `os.environ` at process start — the required timing, identical to the proven
  forward path.
- **Worker** — `megatron_policy_worker.py` `__init__`:
  1. `assert_te_supports_fp4_backward(self.fp4_cfg)` — fail loudly before build.
  2. Per-token awareness of the **existing** train-only `NVTE_BACKWARD_OVERRIDE`
     mechanism. The worker already captures `NVTE_BACKWARD_OVERRIDE`, strips it
     from the global env, and re-applies it **only inside `train()`** (it is
     incompatible with the inference/logprob forward). When
     `fp4_cfg_wants_per_token_backward` is true we force
     `self._nvte_backward_override = None` **and** pop any inherited value, so
     real FP4 dgrad/wgrad runs and no override is re-applied in train.

Why not do the translation in `setup.py`/`apply_te_precision_config`? It runs
**after** the worker's `__init__` capture/pop, so writing `os.environ` there would
defeat the train-only mechanism. Driver-side `env_vars` + worker awareness reuses
that mechanism instead of fighting it.

## Files changed

| File | Change |
|---|---|
| `pyproject.toml` (committed `475077431`) | TE pin → `cael-ling@690ffea` |
| `uv.lock` | regenerated for the pin (uncommitted) |
| `nemo_rl/models/policy/__init__.py` | `Fp4Config`: `backward`, `per_token_{rht,sr,weight_2d}` |
| `nemo_rl/models/megatron/fp4_env.py` (new) | pure translation + capability gate |
| `nemo_rl/models/policy/lm_policy.py` | apply translation into worker runtime `env_vars` |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` | gate + per-token-aware override capture |
| `tests/unit/models/megatron/test_fp4_env.py` (new) | unit tests for the translation/gate |
| `examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd.yaml` (new) | leg-3 recipe — inherits the 8n4g dapo per-token forward recipe, adds `backward: nvfp4_pertoken` |
| `research/vllm-nvfp4-pertoken/run_dapo_longrun.sh` | new `PRECISION=nvfp4_bwd` mode (te690 image guard) |

`setup.py` is intentionally **unchanged**. `apply_te_precision_config` still only
sets `model_cfg.fp4/fp4_recipe/fp4_param` etc.

## Defaults / backward-compat

- `backward` omitted → no NVTE_* emitted → identical to current behavior.
- Existing `-fp4train` (row-scaled fwd + `NVTE_BACKWARD_OVERRIDE=dequantized`)
  unchanged; the worker's train-only override mechanism is untouched on that path.
- Capability gate only trips when per-token backward is explicitly requested.

## Validation status

- Probe image `nemo-rl-te690ffea-probe.sqsh` (TE `2.18.0.dev0+690ffea4`) built;
  PR #3045 recipe classes import; D1 empirically confirmed. Build required
  `NVTE_WITH_NCCL_EP=0` + `srun --cpus-per-task=128` (see `PROGRESS.md` /
  `te690-nccl-ep-build-blocker`).
- `test_fp4_env.py` passes; the megatron code path loaded live in the probe bake
  (with `fp4_cfg=None`).

## Rollout: the 3-leg A/B (8n4g dapo, `run_dapo_longrun.sh`)

All three legs share the 8n4g dapo-512/20k topology and data; only the precision
delta changes. Launch via `research/vllm-nvfp4-pertoken/run_dapo_longrun.sh`:

1. **row-scaled fwd + dequant bwd** — `PRECISION=nvfp4` (the existing per-token
   forward recipe; runs on the v4 image).
2. **per-token fwd + dequant bwd** — `PRECISION=nvfp4` +
   `EXTRA_ARGS='+policy.megatron_cfg.env_vars.NVTE_NVFP4_PER_TOKEN=1'` (keeps
   `NVTE_BACKWARD_OVERRIDE=dequantized`); needs the te690 image.
3. **per-token fwd + FP4 bwd** — `PRECISION=nvfp4_bwd` (the `-fp4bwd` recipe);
   needs the te690 image (the launcher enforces this).

Then ± `per_token_{rht,sr,weight_2d}`. Judge on `token_mult_prob_error`,
`gen_kl`, `grad_norm`, loss convergence. A 1-step wiring smoke is `PRECISION=nvfp4_bwd
MAX_STEPS=1`.

## Open items

- Commit `uv.lock` (currently uncommitted).
- Decide P3/P4 topology (4n4g `-fp4train` vs an 8n4g dapo port).
- `690ffea` is an unmerged fork HEAD — re-pin if it rebases; probe only.
