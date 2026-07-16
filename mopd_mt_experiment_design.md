# True Multi-Teacher MOPD Experiment: Math + Instruction-Following (design only)

## Context

The MOPD smoke (PR #2780 recipe) is green on this cluster (branch `mopd-recipe-bringup`,
job 14048277, baked image `nemo-rl-mopd-main-2026-07-15.sqsh`) — but it is single-teacher
self-distillation (signal ≈ 0). This experiment (1) demonstrates MOPD's *effectiveness*
(student actually improves) and (2) exercises the *multi-teacher* machinery (>1 teacher:
routing, dedup, per-alias overrides) that upstream only covers in unit tests. Domains:
**math** and **instruction following (IF)** — both already have NeMo Gym resources
servers wired into the recipe's `config_paths`.

## 1. Experiment design

### Models (all Qwen3 family — REQUIRED: teacher scores the student's token ids, so tokenizers must be identical)

| Role | Model | Why |
|---|---|---|
| Student | `Qwen/Qwen3-1.7B` | cached, validated in smoke |
| Math teacher | `Qwen/Qwen3-4B-Thinking-2507` | reasoning/math specialist |
| IF teacher | `Qwen/Qwen3-4B-Instruct-2507` | IFEval-strong, non-thinking specialist |

Both teachers are 4B → no capacity confound between arms; the *only* difference between
single- and multi-teacher arms is **which teacher scores which domain**.
Preflight gate: assert all three tokenizers are byte-identical (compare `tokenizer.json`
sha256) — abort the design if not.

### Data (agent name = routing key, embedded per row as `agent_ref`)

- **Math train**: DAPO-17k (already staged), `agent_ref.name = math_with_judge_simple_agent`.
- **IF train**: `nvidia/Nemotron-RL-instruction_following` (HF, Apache-2.0 — the same
  dataset the Gym IF server declares). Transform each row: `prompt` →
  `responses_create_params.input=[{role:user,...}]`, keep `instruction_id_list` +
  checker `kwargs` fields intact, stamp `agent_ref.name = instruction_following_simple_agent`.
- **Mix**: 8k math + 8k IF, shuffled → `$HF_HOME/mopd_mt_data/train-split.jsonl`
  (new dir — do NOT clobber the validated `nanov3_data`).
- **Val**: AIME24 (30 rows, math agent) + 64 held-out IF rows (IF agent) →
  `val-split.jsonl`. Per-agent val rewards are logged separately by the existing
  per-agent metrics in `nemo_rl/experience/rollouts.py` (agent_to_results block) — no
  code needed for per-domain eval.

Extend `tools/prepare_mopd_gym_data.py` → `tools/prepare_mopd_mt_data.py` (same
pattern; adds the IF transform + mixing + held-out split).

### Arms (4; only the teacher map differs — everything else identical)

| Arm | teacher_model_by_agent_name | Nodes | Purpose |
|---|---|---|---|
| 0 base | — (no training) | — | step-0 eval via `val_at_start=true` on Arm 3 (free) |
| 1 single-Thinking | `default_teacher: Thinking-4B` (both agents unmapped → fallback alias) | 3 | best-math single teacher |
| 2 single-Instruct | `default_teacher: Instruct-4B` | 3 | best-IF single teacher |
| 3 multi-teacher | `math_with_judge_simple_agent: Thinking-4B`, `instruction_following_simple_agent: Instruct-4B` | 4 (2 teacher nodes) | MOPD's actual value proposition |

**Hypotheses.** H1: each single-teacher arm improves the student in its teacher's strong
domain and underperforms the other arm in the opposite domain (specialist trade-off).
H2 (headline): the multi-teacher arm matches or beats *each* single-teacher arm in *its*
strong domain simultaneously — per-domain-best distillation without a bigger teacher.

### Multi-teacher feature validation (cheap 5-step smokes before the long runs)

| Test | Config | Expected observable |
|---|---|---|
| F1 routing | Arm 3 config | `[teacher_logprob] group=<alias>` lines for BOTH aliases, sample counts ≈ domain mix in batch |
| F2 dedup | both agents mapped to the SAME checkpoint, `deduplicate_shared_teacher_checkpoints: true` | exactly ONE `✓ Teacher '<primary>' cluster` line / one node reserved |
| F3 per-alias overrides | `non_colocated_teachers.teacher_overrides.<IF alias>.micro_batch_size=4` (default 1) | override visible in teacher worker init log |
| F4 fail-fast | `strict_agent_name_match: true` + one agent unmapped | setup-time ValueError, job dies fast (negative test) |

F1 comes free inside Arm 3's first steps; F2–F4 are three 5-step smokes (~10 min each on
3–4 nodes).

### Training config (per arm; CLI overrides on a single new derived recipe)

One new yaml `examples/configs/recipes/llm/mopd-mt-qwen3-1.7b.yaml` with
`defaults: mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`, changing only: data paths →
`mopd_mt_data`, `max_num_steps: 150`, `val_period: 25`, `val_at_start: true`,
`num_prompts_per_step: 32`, `num_generations_per_prompt: 4`, checkpointing **enabled**
(`save_period: 25` — long-run rule: ckpt-resume must be validated before the long runs),
`cluster.num_nodes: 4`, and the Arm-3 teacher map as the yaml default. Arms 1/2 are pure
CLI overrides (dotted-key dict entries CAN be overridden/added; the base map stays
minimal so no key deletion is ever needed): wandb `project=mopd`,
`name=mopd-mt-arm{1,2,3}` (one shared project per the A/B convention).
Teacher resourcing: recipe defaults (TP2, 1n8g, bf16) per teacher.

### Metrics & analysis

- **Primary**: per-agent val reward vs step (math = math-verify accuracy on AIME24;
  IF = checker pass rate), steps {0,25,…,150} → 2 domains × 4 arms table, deltas vs Arm 0.
- **Distillation health**: `on_policy_distillation/teacher_student_logprob_gap_mean`
  (expect > 0 at start, monotically shrinking), `adv_mean/std`, `teacher_logprob` timing.
- **Guards** (every arm): `train/token_mult_prob_error < 1.1`, `gen_kl` sane, grad_norm
  finite; any violation → stop and debug before burning nodes.
- Analysis via wandb export (`cpu_datamover` node) + `/wandb-analysis` flow.

## 2. Step-by-step execution instructions (for the future execution session)

All steps in worktree `nemo-rl-mopd` (branch `mopd-recipe-bringup`), image
`nemo-rl-mopd-main-2026-07-15.sqsh`. Known traps are already encoded in
`submit_mopd_smoke.sh` / `mopd_bringup_notes.md` (no `UV_CACHE_DIR_OVERRIDE`; NemoGym
venv purge; env-prefix contiguity; `--recursive` submodules).

1. **Stage teachers** (cpu_datamover srun, HF_TOKEN from `~/.env`):
   `snapshot_download` both 4B teachers into `$HF_HOME`; verify safetensors complete.
2. **Tokenizer identity gate**: compare sha256 of `tokenizer.json` across the 3 models
   inside the container; abort if any differ.
3. **Build data**: write + run `tools/prepare_mopd_mt_data.py` on cpu_datamover →
   `$HF_HOME/mopd_mt_data/{train,val}-split.jsonl`; spot-check one row per agent
   (`agent_ref`, `responses_create_params`, IF checker fields present).
4. **Add recipe yaml** `mopd-mt-qwen3-1.7b.yaml` (Arm-3 defaults, as above); derive a
   `submit_mopd_mt.sh` from `submit_mopd_smoke.sh` (param: ARM, extra overrides;
   4 nodes; `--time=12:00:00`). Commit both.
5. **Feature smokes** (5 steps each): F2 dedup, F3 override, F4 strict fail-fast —
   assert the expected log lines (grep), then scancel-cleanup. ~30 min total.
6. **Ckpt-resume smoke**: run Arm 3 config `max_num_steps=6, save_period=3`, kill,
   resume, confirm step 4 continues from ckpt (long-run prerequisite rule).
7. **Launch arms**: submit Arm 3 first (validates F1 routing live), then Arms 1 and 2
   (3 nodes each; parallel if queue allows, else sequential). Present job-parameter
   summary (model paths, wandb names, ckpt dirs, thresholds) before each submit.
8. **Monitor**: job-monitor agent per job; guard metrics above; auto-resubmit on the
   known transient (per-node DNS flake during gym venv build).
9. **Analyze**: wandb export → per-domain reward curves + final table; verdict on H1/H2;
   write results into `mopd_bringup_notes.md` (or a new `mopd_mt_results.md`); commit.
10. **Report**: summary with the 2×4 table, feature-validation checklist (F1–F4), and
    any upstream findings.

## Verification (definition of done for the experiment itself)

- F1–F4 all observed as specified (multi-teacher feature validated beyond unit tests).
- All 3 training arms complete 150 steps with guards green and per-agent val curves logged.
- H1/H2 answered by the 2 domains × 4 arms table (whatever the outcome — a negative
  result on H2 is still a valid feature+effectiveness evaluation).

## Cost estimate

3 smokes (~0.5 h × 3–4 nodes) + resume smoke (~0.5 h) + 3 arms × ~8–12 h × (3,3,4) nodes
≈ 100–130 node-hours on `batch`. Teachers stay bf16 unquantized (MOPD constraint).

## Open risk notes

- Qwen3-*-2507 checkpoints must load under transformers 5.8.1 / vLLM 0.20 in the baked
  image — same `Qwen3ForCausalLM` arch, expected fine; surfaces at step 1–2 if not.
- IF server `/run` contract: verify the transformed rows satisfy it in the F-smokes
  (one full IF rollout with nonzero reward) before long runs.
- Thinking-teacher on non-thinking-student tokens is fine by construction (teacher only
  computes logprobs of the student's sampled tokens).
