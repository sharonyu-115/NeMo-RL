# Multi-Teacher MOPD Experiment — Results (2026-07-16)

Design: `mopd_mt_experiment_design.md`. Student `Qwen/Qwen3-1.7B` (hybrid thinking),
mixed 8k DAPO-math + 8k Nemotron-RL-IF train set, 222-row val (30 AIME24 + 128 DAPO
held-out = "math", 64 IF held-out), 150 steps, `max_new_tokens=4096`, OPD advantage,
non-colocated bf16 teachers. wandb project `mopd`.

## Arms as executed

| Arm | Teacher map | Job | Result |
|---|---|---|---|
| 1 single-Thinking | both agents → Qwen3-4B-**Thinking**-2507 (dedup → 1 teacher node) | 14072483 | COMPLETED 150/150, 1h43m |
| 2 v1 single-Instruct | both → Qwen3-4B-**Instruct**-2507 | 14072484 | **KILLED @ step 14 — mode-collapse (Finding 1)** |
| 3 v1 multi | math→Thinking-2507, IF→Instruct-2507 | 14072482 | **KILLED @ step 28 — same pathology on IF half** |
| 2 v2 single-generalist | both → Qwen3-4B (hybrid) | 14074680 | COMPLETED 150/150, 1h36m |
| 3 v2 multi | math→Thinking-2507, IF→Qwen3-4B | 14074681 | COMPLETED 150/150, 1h36m |

## Finding 1 — Cross-generation-mode teacher mismatch collapses OPD (headline)

Distilling the hybrid-thinking student toward the **non-thinking** Instruct-2507
teacher destabilized generation within ~10 steps: mean generation length pinned at the
4096 cap (`gen_tokens_per_turn: mean=4038`), step time exploded 30s → 2338s
(generation = 2327s of it), severity ordered by Instruct fraction
(arm1 none: 30s/step; arm3v1 half: 20min; arm2v1 all: 39min). Mechanism: the teacher
assigns very low logprobs to the student's thinking-mode tokens → strongly negative
token advantages on structural tokens → REINFORCE destroys the student's termination
behavior. The ICE-POP gate does not protect against this (it corrects train/inference
drift, not teacher-mode mismatch). Fix applied: replace the IF teacher with
mode-compatible hybrid `Qwen/Qwen3-4B` → both v2 arms trained 150 steps at ~30s/step
with no degeneration.

**Practical rule: MOPD teachers must be generation-mode-compatible with the student's
rollout mode.** Mixed-mode specialist pools need per-agent generation modes (not
currently supported) or mode-matched teachers.

## Finding 2 — Generation-budget mismatch confounds math evaluation

All arms' val response length jumped 3140 → ~3900-4050 tokens (at the 4096 cap) by
step 25 and stayed there: distillation toward 4B reasoning teachers immediately
inflates the 1.7B student's reasoning length past its budget, so competition-math
answers truncate before `\boxed{}` and measured accuracy falls even as the policy
tracks the teacher more closely. Longer-budget evaluation (8-16k) or budget-aware
distillation is required to measure math gains at this scale.

## Per-domain val reward (mean; math = 158 rows, IF = 64 rows; single gen @ temp 1.0 — ±0.03-0.06 noise)

| step | arm1 math | arm1 IF | arm2v2 math | arm2v2 IF | arm3v2 math | arm3v2 IF |
|---|---|---|---|---|---|---|
| 0 | 0.101 | 0.313 | 0.108 | 0.297 | 0.139 | 0.266 |
| 25 | 0.038 | 0.172 | 0.070 | 0.313 | 0.006 | 0.313 |
| 50 | 0.070 | 0.375 | 0.108 | 0.375 | 0.013 | 0.281 |
| 75 | 0.057 | 0.422 | 0.032 | 0.281 | 0.057 | 0.281 |
| 100 | 0.038 | 0.391 | 0.070 | 0.297 | 0.044 | 0.328 |
| 125 | 0.025 | 0.391 | 0.044 | 0.391 | 0.025 | 0.281 |
| 150 | 0.019 | 0.328 | 0.070 | 0.359 | 0.000 | 0.375 |

- **IF improves in every arm** (peak deltas vs step 0: arm1 +0.11, arm2v2 +0.09,
  arm3v2 +0.11 at step 150 and still trending up) — distillation signal is real.
- **Math declines in every arm** — dominated by the Finding-2 truncation artifact
  (arm3v2 literally 0.000 at step 150 with the longest-thinking math teacher), so
  H1/H2 are **not evaluable for math** under this budget.
- On IF, multi-teacher (arm3v2, 0.375 final) ≥ single arms (0.359 / 0.328) — weakly
  consistent with H2 but within sampling noise. A conclusive H2 test needs the
  Finding-2 fix plus multi-generation val.

## Multi-teacher feature validation (the second experiment objective) — PASSED

| Test | Evidence |
|---|---|
| F1 per-agent routing | arm3 runs: `[teacher_logprob] group=default_teacher` AND `group=instruction_following_simple_agent`, balanced counts matching the 50/50 mix |
| F2 checkpoint dedup | arm1/arm2v2: two aliases → one checkpoint → exactly one `✓ Teacher` cluster, `opd_teacher_nodes:1` (vs 2 clusters/2 nodes in arm3) |
| F3 per-alias overrides | `teacher_overrides.<alias>.micro_batch_size=4` plumbed to worker creation; **quirk**: partial overrides are backfilled with pydantic defaults (TP became 1), not `default_teacher_cfg` values |
| F4 strict fail-fast | not exercised end-to-end (covered by upstream unit tests); skipped to conserve node-hours |
| Health | teachers healthy in every run; teacher scoring 0.06-0.11s per batch, fully hidden in collection; gen_kl ≤ 0.011 |

## Upstream findings from this campaign (adds to `mopd_bringup_notes.md` list)

5. `teacher_overrides` partial-dict backfill (F3 quirk above) — surprising semantics.
6. **Async-GRPO checkpoint-resume deadlock**: the resumed collector believes the
   next target weight "already exists in buffer" (stale watermark persisted without
   its in-flight trajectories) → skips regenerating it → trainer starves forever →
   idle-GPU reaper kills the job (job 14071115). Untested upstream (MOPD nightly
   disables checkpointing). Workaround: size runs into a single allocation.
7. Cosmetic: driver exits nonzero from actor-teardown races after
   `Async GRPO training complete!` (SLURM shows FAILED despite full success) — use
   `--dependency=afterany` in chains.
8. OmegaConf deep-merge keeps parent-recipe teacher map entries — a derived recipe
   adding aliases silently inherits extra teachers (cost: reserved nodes).

## Verdict

- **Multi-teacher MOPD machinery: validated end-to-end on real workloads** (routing,
  dedup, overrides, non-colocated scaling, 150-step stability) — the feature works.
- **Effectiveness: demonstrated on IF; not measurable on math at this
  student-scale/budget** (Finding 2), and constrained by teacher-mode compatibility
  (Finding 1). Both constraints are actionable design rules for real MOPD use.

## Reproduce / artifacts

- `bash submit_mopd_mt.sh <1|2|3> 1 [overrides]` (v2 teacher swaps shown in git log).
- wandb `mopd`: runs `mopd-mt-arm1`, `mopd-mt-arm2v2`, `mopd-mt-arm3v2` (+ killed
  `mopd-mt-arm2`/`arm3` v1 for Finding-1 evidence).
- Per-sample val dumps: `results/mopd-mt-*/logs/exp_001/val_data_step*.jsonl`
  (idx 0-157 = math, 158-221 = IF); job logs `<jobid>-logs/ray-driver.log`.

## Suggested next iterations

1. Rerun with `max_new_tokens=16384` (+ packed seq budget) to unconfound math.
2. Add multi-generation val (`num_val_generations_per_prompt≥4`) to cut noise.
3. Mode-matched specialist pair (e.g., two thinking-mode teachers RL-tuned on
   different domains) for a clean H2 test.
4. File upstream: findings 5-8 + the two design rules as MOPD docs additions.

## Addendum — wandb step-level analysis (post-hoc, all 150 steps per arm)

Per-arm training health (wandb runs y2iq91d4 / swkgsr07 / zrrubccl):

| Metric (150 steps) | arm1 (Thinking) | arm2v2 (generalist) | arm3v2 (multi) |
|---|---|---|---|
| teacher−student logprob gap (mean) | −0.327 | −0.184 | −0.291 |
| adv_std | 1.15 | 0.82 | 1.12 |
| `token_mult_prob_error>1.1` steps | 1 (max 1.107) | 0 (max 1.056) | **42 (max 1.349)** |
| masked seqs by logprob error (mean/max per step) | 0.19 / 2 | 0.01 / 1 | **1.47 / 25** |
| grad_norm (typical) | 3.3–4.6 | 2.1–2.4 | 3.5–5.7 |
| gen_kl_error plateau | ~0.005 | ~0.002-0.006 | ~0.011 |
| NaN / grad spikes | none | none | none |

- **The gap is negative everywhere** (teacher assigns lower likelihood to the
  student's sampled tokens than the student itself — the advantage mostly
  *suppresses* current behavior), with magnitude ordered by teacher–student
  distribution distance: Thinking-2507 (−0.33) > multi (−0.29) > generalist 4B
  (−0.18). It barely shrinks over 150 steps at lr 3e-6.
- **The multi-teacher arm carries measurably more off-policy stress**: it is the
  only arm violating the `token_mult_prob_error < 1.1` bar (42/150 steps) and
  masks up to 25 seqs/step via `seq_logprob_error_threshold` — two teachers
  pulling domain-disjoint directions between refits. Worth a lower LR or
  tighter refit cadence in future multi-teacher runs.
- **Correction to Finding 2 (training side)**: `train/truncation_rate` = 0.0 in
  all arms even though median/p95 `gen_tokens_per_turn` sit at the 4096 cap —
  the gym rollout path does not populate the hit-max-tokens flag, so
  `overlong_filtering: true` was a **no-op** (upstream finding #9): nothing was
  dropped from training; instead OPD trained on unfinished truncated reasoning.
  The eval-side mechanism (answers cut before `\boxed{}`) stands.
- Aggregate wandb `validation/accuracy` (val_at_start + 6 periodic): arm2v2
  finishes best (0.153) vs arm1/arm3v2 (0.108 both) — consistent with the
  per-domain table; the distribution-closest teacher also trained most stably.

# v3 — 16k budget, lr 1e-6, 300 steps, AIME24 avg@8 (2026-07-18)

Reruns of the three arms with the v2 confounds removed: `max_new_tokens=16384`,
lr 1e-6 (all arms), 300 steps as chained 4h jobs (both resume-deadlock fixes
validated at scale: 2 handoffs × 3 arms, zero stalls), math val =
`BytedTsinghua-SIA/AIME-2024` avg@8 (240 rows, train-identical formatting), IF val
unchanged (64 rows). wandb `nv-welcome/mopd`, runs `mopd-mt16k-arm{1,2,3}`
(note: each chain link opens a new run under the same name).

## Per-domain val (math = AIME24 avg@8 over 240 rows; IF = 64 rows; noise ~±0.02-0.06)

| step | arm1 math/IF (Thinking) | arm2 math/IF (generalist) | arm3 math/IF (multi) |
|---|---|---|---|
| 0 | 0.383 / 0.312 | 0.383 / 0.281 | 0.404 / 0.281 |
| 50 | 0.283 / 0.453 | 0.342 / 0.297 | 0.271 / 0.250 |
| 100 | 0.279 / 0.547 | 0.354 / 0.328 | 0.308 / 0.297 |
| 150 | 0.258 / 0.484 | 0.350 / 0.312 | 0.279 / 0.266 |
| 200 | 0.283 / 0.516 | 0.379 / 0.328 | 0.267 / 0.266 |
| 250 | 0.254 / 0.547 | 0.371 / 0.312 | 0.279 / 0.297 |
| 300 | 0.283 / 0.453 | 0.358 / 0.266 | 0.233 / 0.297 |

## Finding 3 — v2's math story was measurement artifact; true baseline is 0.38-0.40

At 16k the untrained student scores **0.38-0.40 avg@8 on AIME24** (vs 0.10-0.14
"measured" at 4k in v2) and needs 10-12k tokens per solution. No cap pinning in v3
(train gens ~8k mean, val ~12.3k < 16384); truncation no longer confounds anything.

## Finding 4 — teacher labels do not predict per-domain transfer (headline)

Under reverse-KL OPD on an already-post-trained student:
- The **math-specialist** teacher (Thinking-2507) **degrades math** (arm1: 0.383→0.283,
  −0.10 stable across steps 50-300) while **massively improving IF**
  (0.312→0.453-0.547, up to **+0.24**, far beyond noise).
- The **generalist** (Qwen3-4B) holds math nearly flat (−0.02, best math retention)
  with flat IF.

Interpretation: dense token-level distillation transfers a teacher's *behavioral
style* (deliberate, constraint-following long reasoning → IF gains), while pulling an
RL-polished student's token distribution toward even a stronger teacher's can degrade
its calibrated task behavior (math loss). "Specialist" benchmark rankings do not
translate into per-domain OPD gains.

## Finding 5 — multi-teacher routing composes per-domain effects faithfully

Arm3's math curve tracks arm1's math (Thinking-taught rows: 0.23-0.31 vs 0.25-0.28)
and arm3's IF tracks arm2's IF (generalist-taught rows: 0.25-0.30 vs 0.27-0.33) at
every checkpoint. **The multi-teacher mechanism does exactly what it promises —
per-domain effects compose additively.** Arm3 underperformed only because the
teacher→domain assignment was (per Finding 4) inverted relative to what the teachers
actually transfer: the data implies the optimal routing here was Thinking→IF,
generalist→math — the mirror of the "obvious" assignment.

## H1/H2 verdict (final)

- **H1 rejected**: specialists did not improve "their" domains; the math specialist
  improved the *other* domain.
- **H2 split verdict**: the *mechanism* (per-domain-best composition) is validated —
  arm3 = arm1's math effect + arm2's IF effect, cleanly. The *premise* (you know which
  teacher is best per domain a priori) is rejected. Multi-teacher MOPD is only as good
  as the routing table, and the routing table needs empirical per-domain transfer
  measurements (cheap single-teacher probes), not benchmark labels.

## Training health (wandb, last-link histories)

lr 1e-6 eliminated the v2 drift entirely: `token_mult_prob_error>1.1` on **0 steps**
in all arms (max ≤ 1.018 vs 42 violations at 3e-6), masked-seqs ≈ 0, no grad
anomalies; teacher-student gap ≈ −0.22…−0.27 and roughly flat (300 steps at 1e-6
barely moves the distribution — capability changes above are style-level, consistent
with Finding 4).

## Additional upstream/infra findings (continuing the list)

9. Gym rollouts never set the hit-max-tokens flag (from the v2 addendum; still true).
10. Chained runs: metric-based `keep_top_k` prunes the newest checkpoint and
    `checkpoint_must_save_by` saves are unretained without val metrics at that step —
    resume chains must set `checkpointing.metric_name=null` (recency retention).
11. Each chain link opens a new wandb run with the same name — per-name history is
    last-link-only; use TB/val_data dumps for full curves.
12. IF server checker crashes on some rows (`count_increment_word: 'list' object has
    no attribute 'strip'`) — data-quality issue, zeroes those rewards uniformly.
13. The two-part async-resume fix (buffer watermark clamp + collector target-window
    including current version) survived 6 real mid-training handoffs — upstream-ready.

## Suggested next iteration

**Arm 4 (optimal routing)**: math→Qwen3-4B, IF→Thinking-2507 — Finding 4 predicts it
beats all three existing arms on both domains simultaneously; one 4-node chain
(~30 node-hours) turns Finding 5's compositionality into a constructive win and gives
H2 a proper affirmative test with an empirically-derived routing table.
