# Tracker: NVFP4 per-token BACKWARD experiments

Living document. Status snapshot below is **as of 2026-07-28 ~23:20 local**;
refresh with the commands in [§Refresh](#refresh) before trusting the step counts.

Scope: the DAPO-512 / 20k / Qwen3-30B-A3B-Base 8n4g campaign legs that exercise
**real FP4 per-token dgrad/wgrad** (TransformerEngine PR #3045), plus the
forward-only legs that serve as their controls.

Companion docs — do not duplicate their content here:
- `SUMMARY-te-nvfp4-pertoken-backward.md` — implementation/commit series, build blockers.
- `DESIGN-te-nvfp4-pertoken-backward.md` — mechanism (how `backward` maps to NVTE_*).
- `DESIGN-te-nvfp4-pertoken-weight2d.md` — weight-geometry legs (w2d / w1d).
- `ROW-SCALED-vs-FULL-PER-TOKEN.md`, `ANALYSIS-rowscaled-vs-fullpertoken-fwd.md` — forward numerics.
- `session/te-nvfp4-backward/timeline.md`, `session/te-nvfp4-w1d-weight-leg/timeline.md` — chronological run log.

## How to update this doc

- **One row per leg** in [§Registry](#registry). A leg = one recipe = one W&B run
  name = one checkpoint dir. Never reuse a row for a re-launch with different dials.
- **Only deltas go in the per-leg sections.** Everything shared lives in
  [§Shared invariants](#shared-invariants) — if a new leg changes something listed
  there, move that item out of the invariants table rather than silently diverging.
- **Findings are append-only** in [§Findings log](#findings-log), newest last, dated,
  with the evidence (job id / W&B / file:line) that supports them. Do not rewrite an
  earlier finding; add a superseding entry and mark the old one.
- State a leg's verdict only when it is backed by a comparison against a **matched**
  control (see [§Valid comparisons](#valid-comparisons)). Legs that differ in more
  than one dial do not produce verdicts.

---

## Shared invariants

Identical across **every** leg in this tracker unless its row says otherwise.

**Task / topology**: Qwen3-30B-A3B-Base, DAPO-512, 32 prompts x 16 gens/step,
2048 prompt + 20480 response (`max_total_sequence_length: 22528`), train TP2/EP8,
vLLM TP1 x 32 colocated engines (`gpu_memory_utilization: 0.5`, CUDA graphs on),
8 nodes x 4 GB200, 5h allocations chained `afterany`, `grpo.max_num_steps: 1500`
baked into the recipe (never passed per-submission — see
`[[megatron-max-steps-resume-assert]]`), W&B project `qwen3-30b-nvfp4` with a
pinned run id per leg.

**Launcher**: `research/vllm-nvfp4-pertoken/run_dapo_longrun.sh`, `PRECISION=nvfp4_bwd`
(+ `RECIPE_OVERRIDE` / `CONTAINER_IMAGE_OVERRIDE` for the non-default legs).
Recipe inheritance: `<leg>.yaml` -> `...-nvfp4-pertoken-fp4bwd.yaml` ->
`...-nvfp4-pertoken.yaml` -> `...-bf16.yaml`.

### Training-side quantization (TE)

All quantized tensors go through `nvte_nvfp4_per_token_quantize` (K1 amax pass +
K2 encode pass): data **e2m1**, inner scale **e4m3**, inner block **1x16**, outer
amax as **per-row `(M,)` + per-col `(K,)` fp32 vectors**. cuBLASLt cannot consume
vector amaxes, so these GEMMs dispatch to `nvfp4_cutlass_per_token_gemm`.
Slots covered (`transformer_engine/pytorch/quantization.py:1725-1729`): forward
`input`/`output`/`weight`, backward `grad_output`/`grad_input`.

| Tensor / direction | Outer scale | Inner scale |
|---|---|---|
| Fwd activation X, rowwise | **per-token** `(M,)` vector | 1x16 e4m3 along K |
| Fwd activation X, columnwise (wgrad operand) | per-input-channel `(K,)` vector | 1x16 e4m3 along tokens |
| Grad output dY, rowwise | **per-token** `(M,)` vector | 1x16 e4m3 |
| dY, columnwise (wgrad operand) | per-output-channel `(N,)` vector | 1x16 e4m3 |
| Weight W | **varies by leg** — see registry | **varies by leg** |

Activations and gradients are **identical in every leg**: `per_token_weight_2d` /
`per_token_weight_per_tensor_1d` gate on the weight slot only
(`quantization.py:1732-1737`; C++ branch `quantizer.cpp:2536`).

- **Forward**: per-token real NVFP4 (`NVTE_NVFP4_PER_TOKEN=1`,
  `NVTE_NVFP4_ROW_SCALED_ACTIVATION=0`).
- **Backward**: real FP4 per-token dgrad + wgrad (`fp4_cfg.backward: nvfp4_pertoken`).
  The inherited `NVTE_BACKWARD_OVERRIDE=dequantized` is cleared in code — the
  confirming `UserWarning` from `lm_policy.py:151` appears in every fp4bwd job log
  and is the cheapest liveness check that FP4 backward actually engaged.
- **RHT off, SR off** in every leg listed here. RHT would apply to the
  fwd-activation + bwd-grad quantizers; SR to the bwd-grad quantizer only
  (`quantization.py:1744-1750`). RHT is off deliberately —
  `[[nvfp4-pertoken-rht-breaks-traingen]]` and the swizzle-layout kernel bug
  isolated at commit `4c21e2062`.
- Row-scaled activation / 2D-activation / 4over6 are **hard-forced off** under
  per-token in the quantizer factory (`quantization.py:1748,1751-1757`). The
  `NVTE_NVFP4_DISABLE_*` env vars in the recipes are belt-and-braces, not the
  mechanism — do not treat them as the dial.
- **Scope**: MoE-expert `linear_fc1`/`fc2` only; attention BF16
  (`te_precision/attn_bf16_mlp_nvfp4.yaml`); layers 0-1 + 44-47 fully BF16 (f2l4).
  Master weights + optimizer BF16 (`fp4_param: false`) — FP4 exists only as GEMM
  operands. Cast is bf16-input-only, requires `M % 128 == 0`.

### Inference-side quantization (vLLM rollout)

Identical in every leg: `nvfp4_pertoken_rollout.enabled: true`, W4A4, MoE-experts-only.

| Tensor | Outer scale | Inner scale |
|---|---|---|
| **Activation** | **per-token fp32, derived at runtime inside FlashInfer `trtllm_fp4_block_scale_moe`**; no `input_scale` tensor exists, and `w13/w2_input_scale` are overwritten with **1.0** so no static scale participates (`nvfp4_pertoken_vllm.py:91-98`) | 1x16 e4m3 (`group_size: 16`, `dynamic: true`) |
| **Weight w13** (fused gate+up, `(E,2N,K)`) | **one fp32 scalar per expert** = `amax/(6*448)`, shared by both halves (`nvfp4_pertoken.py:241-263`) | 1x16 e4m3 |
| **Weight w2** (down_proj) | one fp32 scalar per expert | 1x16 e4m3 |

Weights real-quantized at refit on the Megatron side, RNE onto the E2M1 grid, no
SR, no RHT. `quant_patterns: ["*.experts.*"]` minus the ignore list (lm_head,
`mlp.gate` router, shared_expert, self_attn, embed_tokens, all norms, **and experts
of layers 0/1/44/45/46/47** — mirrors the training f2l4 window). Backend must be
`FLASHINFER_TRTLLM`: `make_nvfp4_moe_kernel` silently drops `per_token_activation`
on any other backend, so the wrapper raises instead (`nvfp4_pertoken_vllm.py:82-90`).

> **Unverified**: FlashInfer's `trtllm_fp4_block_scale_moe` internals were not read
> (it lives in the container venv; the local `src/vllm` checkout at `4d9c61993` predates
> `make_nvfp4_moe_kernel`). The per-token activation claim rests on the NeMo-RL wiring
> above plus the stage-0/B validation in `README.md` (job 2404103).

### Train/gen alignment implied by the above

- **Activations: matched** in every leg — per-token outer + 1x16 e4m3 inner + e2m1,
  RNE, no RHT, both sides.
- **Weights: mismatched, differently per leg** — rollout always uses one scalar per
  fused projection per expert. See each leg's row.
- **Gradients**: training-only, no rollout counterpart.

---

## Registry

Weight-quant column = the **only** quantization dial that varies between these legs.

Weight direction-dependence (`W_row != W_col^T`) is called out because [F-3](#f-3)
identifies it as the discriminating property.

| Leg | Weight outer / inner | Dir-dep? | Replay | Container (TE) | Status | Reached | Learning |
|---|---|---|---|---|---|---|---|
| [`fp4bwd`](#fp4bwd) | vector / 1x16 | **yes** | off | te690ffea (`690ffea4`) | **STOPPED** 07-26 | step 348, ckpt step_340 | **NO** (-0.86 @150) |
| [`fp4bwd-r3`](#fp4bwd-r3) | vector / 1x16 | yes | **on** | te690ffea (`690ffea4`) | **RUNNING** 2445982 | **step 742 / 1500** | yes, peaked +0.10 @280, decaying |
| [`fp4bwd-w2d`](#fp4bwd-w2d) | **scalar / 16x16** | **no** | off | te690-w1d (`25e1fda6`) | **RUNNING** 2456196 | **step 148 / 1500** | yes, -0.04 @150 and climbing |
| [`fp4bwd-w1d`](#fp4bwd-w1d) | **scalar / 1x16** | **yes** | off | te690-w1d (`25e1fda6`) | **FAILED — 0 steps** | never trained | — (**next to run**) |
| [`fp4bwd-r3-rhtsr`](#adjacent-legs) | vector / 1x16 | yes | on | te690ffea | CANCELLED 07-26 | — | — |

Forward-only controls (dequantized backward, same forward + rollout):

| Leg | Status | Reached | Learning |
|---|---|---|---|
| `fp4fwd` (leg 2) | **STALLED** — step unchanged 40+ keeper cycles, 0 jobs | step 467 | yes, peaked +0.15 @290, decayed to -0.20 @467 |
| `fp4fwd-rhtsr` | CANCELLED 07-26 | — | — |

Chains: `fp4bwd` 2445764-2445777 (killed) - `fp4bwd-r3` 8 queued - `fp4bwd-w2d` 4 queued -
`fp4bwd-w1d` 2456201-2456206 (consumed, see [F-2](#f-2)) - `fp4fwd` 2445985-2445998 -
`fp4fwd-rhtsr` 2448532-2448545 - `fp4bwd-r3-rhtsr` 2448546-2448559.

**Code**: all legs bind-mount the live worktree `nemo-rl-te-bwd` at `/opt/nemo-rl`,
branch `shuangy/te-nvfp4-pertoken-backward`. Current HEAD `5de9a63ea`; the `fp4bwd`
baseline ran at an earlier HEAD. Because the mount is live, **a leg's code is
whatever the worktree held while it ran** — pin the commit in the leg section when
recording a finding.

**Containers**:
- `/lustre/fsw/general_sa/shuangy/images/nemo-rl-te690ffea-probe.sqsh` — TE cael-ling fork `690ffea4` (PR #3045).
- `/lustre/fsw/general_sa/shuangy/images/nemo-rl-te690-w1d-probe.sqsh` — TE sharonyu-115 fork `25e1fda6`, branch `shuangy/nvfp4-per-tensor-1d-weight`, off `690ffea4`. Adds `NVTE_NVFP4_PER_TOKEN_WEIGHT_PER_TENSOR_1D`.

---

## Leg detail

### fp4bwd

Baseline: per-token forward + real FP4 per-token backward, stock per-token weights,
no router replay. Recipe `...-nvfp4-pertoken-fp4bwd.yaml`.

- **Weight quant**: per-row `(M,)` / per-col `(K,)` outer amax vector + 1x16 inner —
  direction-*dependent* at both levels, so `W_row != W_col^T` and the backward
  differentiates a different function than the forward ran. Fixed per W ⇒ a **bias**,
  not noise.
- **vs rollout weights**: no counterpart — rollout uses a per-expert scalar.
- **Status**: chain `scancel`ed 2026-07-26 05:26 at step 348 (jobs 2445764-2445766 ran;
  2445767+ cancelled unstarted). Checkpoint `step_340` intact and resumable.
- **Why stopped**: see [F-1](#f-1).

### fp4bwd-r3

`fp4bwd` + `policy.router_replay.enabled: true` (vLLM returns gen-time MoE routing;
the Megatron training forward replays it). Minimal enablement — the vLLM
monolithic-capture fix (`patches.py::_patch_vllm_moe_routed_experts_capture`,
`1d3f0eea8`) means the old r3 sampling/packing workarounds are unnecessary, so DAPO
sampling/packing stay identical to `fp4bwd`. Clean single-dial A/B.

- **Weight quant**: same as `fp4bwd`.
- **Status**: RUNNING, the longest-lived leg. Keeper tops up the chain when queued < 4.

### fp4bwd-w2d

`fp4bwd` + `per_token_weight_2d: true` -> `NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1`.

- **Weight quant**: per-tensor **scalar** outer amax + **16x16** inner tiles. A 16x16
  tile holds the same 256 numbers read either way, so rowwise == columnwise^T —
  verified byte-exact (0.00% mismatch, all shapes) by
  `verify_w1d_weight_legs.py` gate B. Removes the fwd/bwd transposition bias at the
  cost of one scale per 256 elements instead of per 16.
- **vs rollout weights**: the scalar outer **matches** the rollout's per-expert scalar
  (TE's fused `linear_fc1` corresponds to vLLM's fused `w13`); only inner geometry
  differs (16x16 vs 1x16). So this leg also narrows the train/gen weight gap — a
  second effect that must be separated from the bias-removal effect when reading
  its curve.
- **Status**: RUNNING. Rolled 2456195 -> 2456196 at step 134 (5h wall, clean resume).
- **Caveat**: never smoke-tested under real FP4 backward (the handoff budgeted a
  1-step smoke for this leg; it was not run).

### fp4bwd-w1d

`fp4bwd` + `per_token_weight_2d: true` + `per_token_weight_per_tensor_1d: true` ->
scalar outer amax kept, inner tiles back to 1x16. The ablation that separates
inner-block geometry from outer-amax granularity, with `fp4bwd-w2d` as its control.

- **Weight quant**: per-tensor scalar outer + 1x16 inner. Reintroduces the
  direction-dependence by design (gate B: ~76-77% of elements differ rowwise vs
  columnwise^T). Cast is byte-identical to a plain per-tensor 1D weight, so
  `scale_inv` geometry is unchanged and the per-token CUTLASS GEMM needs no work.
- **Status**: **FAILED before training.** All six chained jobs (2456201-2456206) died
  in `setup()` at `wandb.init`, ~18 min each; chain consumed; checkpoint dir empty.
  See [F-2](#f-2). Needs a fresh chain submission.

### Adjacent legs

- **`fp4fwd` (leg 2)** — per-token forward + `backward=dequantized`. The forward-only
  control for the backward legs. **Stalled at step 467** with zero jobs; the chain
  keeper has logged `STUCK ... NOT topping up` every 30 min since ~07-27 03:00 and is
  correctly refusing to resubmit. Undiagnosed.
- **`fp4bwd-r3-rhtsr` / `fp4fwd-rhtsr`** — RHT+SR ablations, both cancelled 2026-07-26
  20:23 after RHT was isolated to a swizzle-layout kernel bug (`4c21e2062`). Do not
  relaunch the RHT legs on this build.

---

## Valid comparisons

| Pair | Dials differing | Verdict-capable? |
|---|---|---|
| **`fp4bwd-w2d` vs `fp4bwd-w1d`** | weight inner geometry only (outer held scalar) | **Yes — the priority run.** Single-dial test of [F-3](#f-3) |
| `fp4bwd` vs `fp4bwd-r3` | router replay only | Yes — replay is *sufficient* to rescue learning ([F-1](#f-1)), but F-3 shows it is not the only fix |
| `fp4bwd-w2d` vs `fp4bwd-w2d-r3` | router replay only, bias already removed | Yes, once w2d-r3 exists — tests additivity |
| `fp4bwd` vs `fp4bwd-w2d` | weight outer **and** inner | Two dials — establishes *that* weight geometry rescues learning, not *which* level does |
| `fp4bwd-r3` vs `fp4bwd-w2d` | router replay **and** weight geometry | **No** — do not compare these curves |
| `fp4fwd` vs `fp4bwd` | backward precision (both replay-off) | Yes — and the gap is the whole problem: forward-only learns, FP4-backward does not |
| `fp4fwd` vs `fp4bwd-w2d` | backward precision **and** fwd weight geometry | Two dials — `per_token_weight_2d` changes the forward weight cast too |
| `fp4fwd-r3` vs `fp4bwd-r3` | backward precision, replay matched | Clean, but `fp4fwd-r3` not yet launched |

---

## Findings log

### F-1 — FP4 per-token backward does not learn without router replay — **SUPERSEDED by [F-3](#f-3)**
*2026-07-26. Evidence: jobs 2445764-2445766 (`fp4bwd`) vs 2445971+ (`fp4bwd-r3`);
resolved-MasterConfig diff; `session/te-nvfp4-backward/timeline.md:126-134`.*

> **Superseded 2026-07-28.** The observation (fp4bwd flat, fp4bwd-r3 learns) is real,
> but the causal attribution to router replay was drawn from a two-leg comparison in
> which replay was the only dial *available at the time*. `fp4bwd-w2d` now learns with
> replay OFF, so replay is sufficient but not necessary. See [F-3](#f-3).

`fp4bwd` reward flat at ~-0.87 over 130 steps; `fp4bwd-r3` climbs to ~-0.40 by step 40.
The resolved-config diff showed the **only** difference is
`policy.router_replay.enabled` (F vs T) plus its two downstream flags
(`enable_return_routed_experts`, `moe_enable_routing_replay`). Interpretation
recorded at the time: without replay the training forward recomputes MoE routing that
diverges from generation routing under NVFP4 quant, so the GRPO importance ratio is
computed against the wrong experts and carries no usable gradient signal. cf.
`[[nvfp4-r2-divergence-rootcause]]`. The `fp4bwd` chain was cancelled as a result.

**Consequence not yet acted on**: `fp4bwd-w2d` and `fp4bwd-w1d` are both **replay-off**,
i.e. running in the regime this finding says does not learn. Decide whether the
weight-geometry ablation should carry `router_replay.enabled: true` before spending
more allocations on it.

### F-2 — w1d chain lost to a wandb.init timeout, not an FP4 problem
*2026-07-27 19:29-20:24. Evidence: jobs 2456201-2456206, all FAILED;
`2456203-logs/ray-driver.log`.*

```
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
  at nemo_rl/utils/logger.py:209 -> wandb.init(**cfg, dir=log_dir)
  via nemo_rl/algorithms/grpo.py:364 -> Logger(logger_config)
```

Every job in the chain failed identically in `setup()` before any training step,
burning ~18 min each until the chain was exhausted. This is transient W&B API
reachability, not the new TE flag — the leg has **never** executed a training step, so
nothing is known about per-tensor-1D weights under FP4 backward. The
`Container/Code Version Mismatch` warning in the same logs is the expected TE re-pin
drift (warning-only), not the cause.

**Action**: resubmit a fresh w1d chain. Consider raising `init_timeout` or making
W&B init non-fatal so an API blip cannot consume a whole dependency chain.

### F-3 — The blocker is the weight transposition bias, not the absence of router replay
*2026-07-28. Evidence: W&B `nv-welcome/qwen3-30b-nvfp4`, `train/reward` +
`train/gen_kl_error`, fetched by `fetch_bwd_curves.py`. Run ids in that script.*

`train/reward` in windowed means, all four legs, at comparable steps:

| Leg | replay | FP4 bwd | weight geometry | reward @~150 | reward peak | gen_kl @~150 |
|---|---|---|---|---|---|---|
| `fp4bwd` | off | yes | vector outer / 1x16 — **direction-dependent** | **-0.855** | -0.715 @346 (still climbing, never positive) | 0.0100 |
| `fp4bwd-r3` | **on** | yes | vector outer / 1x16 — direction-dependent | -0.239 | **+0.102** @250-312 | 0.0042 |
| `fp4bwd-w2d` | off | yes | **scalar outer / 16x16 — direction-INDEPENDENT** | -0.183 | -0.043 @136-162 (still climbing) | 0.0074 |
| `fp4fwd` | off | **no** (dequantized bwd) | n/a in bwd | -0.130 | **+0.154** @273-311 | 0.0053 |

Three legs learn; only `fp4bwd` does not. The two properties that distinguish the
failing leg from each learner:

- vs `fp4bwd-r3` — router replay (the F-1 reading).
- vs `fp4bwd-w2d` — **weight direction-dependence**. w2d is replay-OFF and learns
  ~4x faster than `fp4bwd` at matched steps.
- vs `fp4fwd` — no FP4 backward at all, so no columnwise weight cast and no bias.

Replay is therefore **sufficient but not necessary**. The property common to all
three learners and absent from `fp4bwd` is that none of them differentiates through
a weight for which `W_row != W_col^T`. Mechanism (see
`session/te-nvfp4-weight-quant-legs/handoff.md`): a direction-dependent weight makes
the backward compute the gradient of a different function than the forward ran; the
mismatch is a fixed function of W, so it is a **bias** that does not average out.
`fp4bwd` also carries the highest sustained `gen_kl_error` (~0.010 vs 0.004-0.007),
consistent with a train/gen mismatch rather than plain gradient noise.

**Status: strong but not yet isolated.** `fp4bwd` -> `w2d` moves *two* sub-dials at
once (outer amax vector -> scalar, and inner 1x16 -> 16x16). `w1d` (scalar outer,
1x16 inner, direction-dependent again) is the single-dial discriminator — see
[§Open questions](#open-questions--next-actions) item 1.

**Secondary observation (all learning legs)**: reward peaks then decays while
`gen_kl_error` rises — `fp4fwd` peaks +0.154 @~290 then falls to -0.205 @467 with
gen_kl 0.0069 -> 0.0187 (2.7x); `fp4bwd-r3` peaks +0.102 @~280 then falls to -0.154
@749 with gen_kl 0.0048 -> 0.0084. Common late-run degradation, orthogonal to the
weight-geometry question, not yet investigated. cf. `[[nvfp4-r2-divergence-rootcause]]`.

---

## Open questions / next actions

1. **[NEXT] Resubmit `fp4bwd-w1d`, replay OFF, as the side-by-side for the running
   `fp4bwd-w2d`.** This is now the highest-value experiment in the campaign: it is the
   single-dial discriminator for [F-3](#f-3) (scalar outer amax held fixed; only inner
   geometry 16x16 -> 1x16 changes, restoring direction-dependence on ~76-77% of
   elements per gate B). Keep replay OFF — replay-on would re-confound the comparison
   with w2d and mask the effect under test.
   **Falsifiable prediction**: if the bias mechanism is right, w1d tracks `fp4bwd`
   (reward ~-0.85 at step 150), not `w2d` (~-0.18). If w1d instead tracks `w2d`, the
   fix was the *outer amax granularity*, not direction-independence, and F-3's
   mechanism is wrong.
2. **Then `fp4bwd-w2d-r3`** (w2d + `router_replay.enabled: true`; new 3-line recipe) —
   are the two fixes redundant or additive? Decides what the shipping recipe should be.
   Runs against `fp4bwd-w2d` (replay dial) and `fp4bwd-r3` (weight dial).
3. **`fp4fwd-r3`** — matched-replay forward control for `fp4bwd-r3`, quantifying what
   real FP4 backward costs vs dequantized backward. Recipe does not exist yet
   (`fp4fwd` + `router_replay.enabled: true`). Demoted below 1-2: `fp4fwd` (replay-off,
   step 467) already gives a usable forward reference.
4. **Late-run reward decay + rising gen_kl** in every learning leg (F-3 secondary).
   Affects any conclusion drawn past ~step 300.
5. **Diagnose `fp4fwd` (leg 2)** stalled at step 467 for ~20h — it is also the leg that
   shows the decay most clearly.
6. **Smokes never run**: neither `WEIGHT_2D=1` nor `WEIGHT_PER_TENSOR_1D=1` had a
   1-step smoke under real FP4 backward. w2d has now de-risked WEIGHT_2D empirically
   (148 steps, learning); w1d's flag is still unexercised in a real run.
7. **Leg identifiability**: TE's `_make_repr` prints `per_token_weight_2d=True` for the
   w1d leg too and never mentions the new flag. Log the resolved
   `NVTE_NVFP4_PER_TOKEN_*` set from the NeMo-RL side, or w1d/w2d runs are
   indistinguishable after the fact. **Do this before launching w1d** — it is the one
   run where mistaking leg C for leg B would silently produce a duplicate of w2d.
8. **Unverified premise**: FlashInfer's per-token activation kernel internals (see the
   note in [§Shared invariants](#inference-side-quantization-vllm-rollout)).

---

## Refresh

```bash
cd /lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd

# what is alive
squeue -u $USER -o "%.10i %.40j %.8T %.10M %R"

# current step of a running leg
grep -a "= Step " <jobid>-logs/ray-driver.log | tail -1

# resolved config header (precision / recipe / container / wandb id) for a job
sed -n '1,12p' logs/<name>-<jobid>.out

# did FP4 per-token backward actually engage?
grep -a "backward='nvfp4_pertoken' requires NVTE_BACKWARD_OVERRIDE UNSET" logs/*<jobid>*.err

# chain health
tail -20 session/te-nvfp4-backward/chain_keeper.log

# post-mortem on a dead chain
sacct -j <id1>,<id2>,... -X -o JobID,JobName%36,State,Start,End,Elapsed
```
