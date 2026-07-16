# Multi-Teacher On-Policy Distillation (MOPD) in NeMo-RL — Functional Design and Flow Charts

Analysis of upstream `main` (`be8e7e568`, 2026-07-15). MOPD landed in a single PR:
**`5c2322cfc` — "feat: Multi-Teacher On-Policy Distillation (MOPD) (#2780)"**, implementing the
MiMo-V2-Flash paradigm ([arXiv:2601.02780](https://arxiv.org/abs/2601.02780)).

Distinct from two other distillation paths in the repo:

- Classic on-policy logit distillation (`examples/run_distillation.py`) — forward-KL over the full vocabulary.
- Cross-tokenizer **off**-policy multi-teacher distillation (`examples/run_xtoken_off_policy_distillation.py`).

---

## 1. Core function

MOPD distills one or more frozen teachers into the student policy by running **async GRPO with
NeMo Gym rollouts** and replacing the reward-based advantage with a token-level distillation
advantage:

```
Â_t = sg[ log π_teacher(t) − log π_student(t) ]     (Eq. 8, MiMo-V2-Flash)
```

- Enabled with `grpo.adv_estimator.name: opd` plus a top-level `on_policy_distillation:` block;
  requires `grpo.async_grpo.enabled: true` and NeMo Gym rollouts
  (entrypoint `examples/nemo_gym/run_grpo_nemo_gym.py`).
- Maximizing this advantage is reverse-KL minimization toward the teacher; it only needs the
  teacher's logprob of the **sampled** token — no full-vocab logits transfer.
- "Multi-teacher" means **routing, not ensembling**: `teacher_model_by_agent_name` maps each
  NeMo Gym agent name to a teacher checkpoint; each sample is scored by exactly one teacher.
  Unmapped agents fall back to `default_teacher_alias` (or raise with
  `strict_agent_name_match: true`).
- Distillation is the entire learning signal — `reference_policy_kl_penalty: 0.0`, no reward term.

## 2. High-level design

- **Rides the GRPO trainer unchanged.** Only the advantage estimator is swapped; the OPD advantage
  flows through the ordinary `advantages` key.
- **Teacher scoring happens at collection time.** The async trajectory collector scores each
  finished rollout batch and attaches `teacher_reference_logprobs` before it reaches the replay
  buffer, so scoring overlaps with generation.
- **Non-colocated teachers.** Each distinct teacher checkpoint gets its own `RayVirtualCluster`
  on dedicated nodes reserved out of the policy's node budget (`grpo.py:634-655`; setup fails if
  nothing is left for the policy). Placement is NVLink-domain-aware. Reference 3-node recipe =
  1 policy + 1 vLLM generation + 1 teacher node.
- **Teachers are Megatron inference-only.** DTensor teacher config is rejected, PEFT/draft modules
  are stripped, quantization is ignored with a warning; the worker group exposes only
  `get_logprobs()`.
- **Dedup.** Aliases sharing one checkpoint are collapsed onto a single worker group
  (`deduplicate_shared_teacher_checkpoints: true`).
- **Loss configuration** matches the paper: `disable_ppo_ratio: true` gives the REINFORCE form
  `−Â·log π`; async off-policy drift is corrected by the ICE-POP hard gate
  (`use_importance_sampling_correction: true`, `truncated_importance_sampling_type: icepop`),
  which zeroes tokens whose train/inference importance weight falls outside bounds — the `w_t`
  gate from the paper, kept out of the advantage itself.

## 3. Low-level implementation map

| Concern | Location |
|---|---|
| Config schemas (`OnPolicyDistillationConfig`, `NonColocatedTeachersConfig`, `TeacherResourceConfig`), routing (`resolve_reference_aliases`), teacher WG creation, fail-fast `assert_prev_logprobs_available` | `nemo_rl/algorithms/opd.py` |
| `OPDAdvantageEstimator.compute_advantage`: `(teacher_logprobs - prev_logprobs).detach() * mask`, no normalization | `nemo_rl/algorithms/advantage_estimator.py:514-590` |
| Dispatch `elif name == "opd"` | `nemo_rl/algorithms/grpo.py:1971-2002` |
| Teacher scoring in `AsyncTrajectoryCollector`: group by alias→group, per-teacher lock + ThreadPoolExecutor across teachers, pad to teacher DP multiple, `twg.get_logprobs()` → `teacher_reference_logprobs` | `nemo_rl/algorithms/async_utils/trajectory_collector.py:651-838` |
| `TeacherWorkerGroup` (inference-only mcore, `get_logprobs` only; DTensor/PEFT/quant guards) + `create_teacher_configs_from_opd_config` dedup/overrides | `nemo_rl/models/policy/teacher_worker_group.py` |
| Node reservation from policy budget; WG creation after policy/vLLM init (HF→mcore cache race) | `nemo_rl/algorithms/grpo.py:634-655, 1270-1284` |
| `agent_ref` attached to samples (NeMo Gym rollout path only) | `nemo_rl/experience/rollouts.py:1947-1997` |
| Advantage consumption in train loop (`_pad_teacher_logprobs`, `train_data["advantages"]`) | `nemo_rl/algorithms/grpo.py:4004-4222` |
| REINFORCE form + ICE-POP token gate | `nemo_rl/algorithms/loss/loss_functions.py:381-620` |
| Doc / recipe | `docs/about/algorithms/mopd.md`, `examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml` |

Fail-fast: `adv_estimator: opd` raises at setup if `loss_fn.force_on_policy_ratio: true` with no
`grpo.seq_logprob_error_threshold` would zero `prev_logprobs` (advantage would silently degrade to
`teacher_logprobs − 0`).

Minor gap: `get_teacher_routing_metrics()` in `opd.py` is defined but has no live call site on
main — the routing diagnostics it computes are not logged yet.

---

## 4. Flow charts

Reference layout = the 3-node recipe `mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`.

### 4.1 Placement — who lives on which GPUs

Three disjoint `RayVirtualCluster`s (nothing is colocated); the collector and replay buffer are
CPU-side Ray actors; NeMo Gym talks to the student's vLLM engine over HTTP.

```
                        ┌─────────────────────────────────────────────┐
                        │  Driver (grpo.py async train loop, CPU)     │
                        │  ├─ AsyncTrajectoryCollector (Ray actor)    │
                        │  ├─ ReplayBuffer            (Ray actor)     │
                        │  └─ NeMo Gym agent/env loop (HTTP client)   │
                        └───────┬──────────────┬──────────────┬───────┘
                                │              │              │
        ┌───────────────────────▼──┐  ┌────────▼───────────┐  ┌▼──────────────────────────┐
        │ Node 0 · 8 GPUs          │  │ Node 1 · 8 GPUs    │  │ Node 2 · 8 GPUs           │
        │ STUDENT TRAINING         │  │ STUDENT GENERATION │  │ TEACHER "default_teacher" │
        │ (Megatron policy,        │  │ (vLLM async engine │  │ (TeacherWorkerGroup =     │
        │  optimizer, trainable)   │  │  + HTTP server,    │  │  Megatron inference-only: │
        │                          │  │  frozen weights,   │  │  no optimizer, no ref     │
        │ cluster: "policy"        │  │  version k)        │  │  model, get_logprobs only)│
        └──────────▲───────────────┘  └────────▲───────────┘  │ cluster: "teacher_<alias>"│
                   │ refit: new weights k+1 ───┘              └───────────────────────────┘
                   │ (policy → vLLM, every train step)

        More teachers?  each distinct checkpoint = +1 cluster "teacher_<alias>" on its own
        nodes (reserved from the policy budget, grpo.py:634-655); aliases sharing a
        checkpoint are deduped onto ONE cluster.
```

### 4.2 Scheduling — the two concurrent control loops

The collector runs continuously; the trainer consumes. The only synchronization points are the
replay-buffer barrier and the refit pause.

```
 COLLECTOR ACTOR (runs forever)                    TRAINER LOOP (driver, step k)
 ══════════════════════════════                    ═══════════════════════════════
 ┌─► acquire in-flight slot                        ┌─► wait until ReplayBuffer has a
 │   (semaphore = num_prompts_per_step             │   complete batch for step k
 │    × max_trajectory_age_steps)                  │   (only trajectories with
 │        │                                        │    age ≤ max_trajectory_age_steps)
 │   spawn rollout thread                          │        │
 │        │                                        │   compute prev_logprobs (student,
 │   NeMo Gym multi-turn rollout                   │   training engine, fprop only)
 │   via student vLLM HTTP server                  │        │
 │        │                                        │   advantages = teacher_lp − prev_lp
 │   batch done → group samples by teacher         │        │
 │        │                                        │   loss (REINFORCE + ICE-POP gate)
 │   ┌ per-teacher lock ┐  ← serializes calls      │   optimizer step → weights k+1
 │   │ teacherA.get_    │    to ONE teacher        │        │
 │   │ logprobs()       │    (NCCL desync guard);  │   pause collector  ──────────────┐
 │   └──────────────────┘    DIFFERENT teachers    │   refit vLLM ← policy weights    │ barrier
 │        │                  run in parallel       │   resume collector ◄─────────────┘
 │   push scored batch to ReplayBuffer             │   collector.current_weight_version = k+1
 └──(tagged with weight version)                   └─── k += 1
```

### 4.3 Data flow — what keys are produced where

```
 dataset prompt
      │
      ▼
 NeMo Gym rollout (agent loop ↔ student vLLM HTTP)          rollouts.py:1947-1997
      │  produces per sample:
      │    message_log (tokens, roles)        generation_logprobs (vLLM, per token)
      │    agent_ref = {"name": <agent>}      loss_mask (assistant tokens only)
      ▼
 AsyncTrajectoryCollector                                   trajectory_collector.py:651-838
      │  agent_ref.name ──resolve_reference_aliases()──► teacher alias ──dedup──► group
      │  flatten message_log → input_ids [B,S], pad to teacher DP multiple
      │  TeacherWorkerGroup.get_logprobs(input_ids)
      ▼
      + teacher_reference_logprobs [B,S]   ◄── NEW KEY added to the batch
      │
      ▼
 ReplayBuffer ──sample──► trainer batch                      grpo.py:4004-4222
      │
      ▼
 student training engine fprop → prev_logprobs [B,S]
      │
      ▼
 OPDAdvantageEstimator:                                      advantage_estimator.py:514
      advantages = (teacher_reference_logprobs − prev_logprobs).detach() × mask
      │
      ▼
 ClippedPGLossFn:                                            loss_functions.py:381-620
      loss  = −advantages · curr_logprobs          (disable_ppo_ratio → REINFORCE)
      gate  : zero tokens where exp(prev_lp − generation_lp) ∉ [ε_low, ε_high]  (ICE-POP)
      │
      ▼
 optimizer step → new student weights → refit to vLLM
```

### 4.4 Overlap — how the three stages pipeline in wall-clock time

Generation, teacher scoring, and training for *different* batches run simultaneously on their own
GPUs; the refit pause is the only global stall. Staleness is bounded by
`max_trajectory_age_steps` (A) and corrected by the ICE-POP gate.

```
 time ─────────────────────────────────────────────────────────────────────────►

 GEN GPUs      │ gen B1 ░ gen B2 ░ gen B3 ░ gen B4 ▐▌ gen B5 ░ gen B6 ▐▌ gen B7
 (vLLM, node 1)│                                   ▐▌refit             ▐▌refit
               │                                   ▐▌pause             ▐▌pause
 TEACHER GPUs  │        score B1   score B2   score B3    score B4   score B5
 (node 2)      │        (teacher A ∥ teacher B — different teachers concurrent,
               │         same teacher serialized by its lock)

 POLICY GPUs   │ idle…      train step k        train step k+1      train step k+2
 (node 0)      │            (consumes B1..)     (consumes B2,B3..)  (consumes B4..)
               │                        └─opt──►▐▌                          ▐▌
               │                                ▐▌ weights k+1 ──► vLLM     ▐▌ k+2 ──► vLLM

 in flight at once: up to num_prompts_per_step × A prompt groups
 e.g. while the policy trains step k on batches generated under weights ≤ k−1,
 the collector is ALREADY generating + teacher-scoring batches for steps k+1…k+A.
```

Two design points that make the overlap safe:

- **Teacher scoring sits on the collector side of the buffer**, so its latency hides inside
  generation wall-clock instead of extending the train step — this is why teachers must be
  non-colocated (time-sharing GPUs with the policy would serialize the pipeline).
- **Off-policyness from the overlap** (a trajectory generated under weights k−2 trained under
  weights k) is bounded by the replay buffer's age eviction and corrected per token by the
  ICE-POP importance-weight gate in the loss, rather than by PPO ratio clipping.

---

## 5. Q&A findings

### Can MOPD run without NeMo Gym (single-turn)?

**No, not on current main — but single-turn *with* NeMo Gym works fine.**

There is no explicit "OPD requires NeMo Gym" assertion (the only hard assert: non-colocated
teachers require async GRPO, `grpo.py:637`). The dependency is structural, through `agent_ref`:

- Teacher scoring is gated on
  `if self._has_distillation_teachers and "agent_ref" in final_batch_cpu:`
  (`trajectory_collector.py:813`).
- The **only** producer of `agent_ref` in the rollout data path is the NeMo Gym rollout
  (`rollouts.py:1947`). The plain async rollout path never attaches it.

With `should_use_nemo_gym: false`: teacher scoring is **silently skipped**, the batch reaches the
trainer without `teacher_reference_logprobs`, and the first train step crashes with
`ValueError("OPD requires teacher_logprobs")` from `OPDAdvantageEstimator`. No setup-time check
covers this path.

Related notes:

- **Single-turn is not the obstacle** — NeMo Gym controls the turn count from its agent/env
  config; single-turn tasks under NeMo Gym are the normal case. The non-Gym rollout path is
  what's unsupported, because agent-name→teacher routing is the teacher-selection mechanism and
  only Gym samples carry an agent name.
- **Colocated teachers don't exist.** Teacher worker groups are only created when
  `non_colocated_teachers.enabled: true`; OPD-enabled without it hits the same late crash.
  Non-colocated + NeMo Gym + async GRPO is the only supported combination.
- A single-teacher non-Gym mode would be a small extension (default-route samples when
  `agent_ref` is absent), but no such fallback exists on main.

### What recipe did PR #2780 ship? Is it actually multi-teacher?

One recipe plus its nightly driver:

- `examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`
- `tests/test_suites/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.sh`

Shape: **self-distillation of `Qwen/Qwen3-1.7B` (student == teacher) on 3 nodes × 8 GPUs** —
1 node Megatron policy with sequence packing, 1 node vLLM generation, 1 node teacher (TP2, bf16,
`micro_batch_size: 1`). Key settings: `async_grpo.enabled: true`, `adv_estimator.name: opd`,
`seq_logprob_error_threshold: 2.0`, `train_global_batch_size: 32`, NeMo Gym env, one alias
(`default_teacher`) behind `default_teacher_alias`. Dataset paths are placeholders that must be
overridden. Since student == teacher, the loss should hover near zero — it is a **correctness
smoke test**, not a demonstration of distillation gains.

**The recipe is single-teacher.** The multi-teacher parts — routing multiple agent names to
different checkpoints, checkpoint dedup, per-alias resource overrides, concurrent scoring — are
exercised only in unit tests on main:

- `tests/unit/algorithms/test_opd.py` (434 lines) — alias resolution, fallback/strict behavior,
  routing metrics
- `tests/unit/models/policy/test_teacher_worker_group.py` — two-teacher configs
  (`math: /ckpt/math`, `code: /ckpt/code`), dedup, overrides

No checked-in end-to-end recipe runs two distinct teachers. (The existing multi-teacher recipe
`distillation-xtoken-off-policy-multiteacher-*.yaml` belongs to the separate cross-tokenizer
off-policy feature, not MOPD.) A real multi-teacher run — multiple NeMo Gym agents, each mapped
to its own teacher checkpoint, with dedicated nodes per teacher — would be user-assembled; the
config surface exists but no reference recipe demonstrates it.
