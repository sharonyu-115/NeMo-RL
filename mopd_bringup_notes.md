# MOPD Recipe Bring-up Notes (branch `mopd-recipe-bringup`)

Goal: make the PR #2780 smoke recipe `mopd-qwen3-1.7b-3n8g-megatron-pack.yaml` runnable
on the SLURM cluster, validating the env for larger MOPD experiments.
**Result (2026-07-15): job 14048277 COMPLETED** — 5/5 async GRPO steps, teacher workers
healthy, per-batch `teacher_logprob` scoring, official nightly checks PASS
(`median(train/loss) = 0.0` < 0.05; `median(train/token_mult_prob_error) = 1.0152` < 1.1).

## How to run

```sh
# one-time: bake the image matching this checkout (1 node, ~50 min)
sbatch submit_env_refresh_mopd.sbatch
# one-time: stage Qwen3-1.7B + gym math jsonl (CPU node)
srun -A coreai_dlalgo_nemorl -p cpu_datamover --container-image <any nemo-rl image> \
  --container-mounts /lustre/fs1:/lustre/fs1 \
  bash -c 'HF_HOME=.../src/NeMo-RL/hf python tools/prepare_mopd_gym_data.py'
# each run:
bash submit_mopd_smoke.sh [extra config overrides...]
```

Assets: image `~/images/nemo-rl-mopd-main-2026-07-15.sqsh`; data
`$HF_HOME/nanov3_data/{train,val}-split.jsonl` (17,398 DAPO-17k train / 30 AIME24 val).

## Failure log (9 iterations) and the traps they encode

| Job | Failure | Root cause / fix |
|---|---|---|
| 14031420 | `megatron-core` metadata error | Nested submodule `Megatron-Bridge/3rdparty/Megatron-LM` empty — init with `--recursive` |
| 14033142 | `ImportError: AutoProcessor ... (unknown location)` | See 14044793 — same root cause (cache shadow), misattributed to image drift at the time |
| 14034886 | raylet loop: `No module named 'ray._private.node'`, hung at `vllm_policy 0/8` | Same root cause; job **hangs** burning 3 nodes rather than failing |
| 14035744 | exit 1 at startup | `source ~/.env` doesn't exist in container (`--no-container-mount-home`); export secrets on submit host |
| 14039023 | — | Bake succeeded: torch 2.11.0+cu130 / transformers 5.8.1 / vllm 0.20.0 / ray 2.55.1, fingerprint stamped |
| 14041655 | Cluster idle, no driver | Comment line mid sbatch env-prefix continuation silently emptied `COMMAND` → ray.sub interactive mode |
| 14042660/14044793 | transformers + ray corruption again, even on fresh image | **Real root cause of the whole class: `UV_CACHE_DIR_OVERRIDE`** — ray.sub bind-mounts it over `/root/.cache/uv`, but image venv packages are *symlinks into that dir* → every symlinked package dangles. Diagnosed offline via `unsquashfs -ll`. Never set this override with baked images. |
| 14046116 | Rollouts stuck, `buffer_size=0`, swallowed `KeyError('agent_ref')` | Gym data rows MUST carry `agent_ref: {"name": <gym agent>, "type": "responses_api_agents"}`; also purge the stale baked NemoGym venv (SETUP_COMMAND) so it rebuilds from the pinned Gym submodule |
| 14048084 | flash-attn fetch: DNS failure on one node | Transient per-node flake during gym venv build — plain resubmit |
| 14048277 | — | **PASSED** |

## Upstream follow-ups identified

1. `ray.sub` `UV_CACHE_DIR_OVERRIDE` corrupts images with symlinked venvs (hang, not fail) — candidate `fix:` PR.
2. `adv_estimator: opd` has no setup-time validation that samples carry `agent_ref` / NeMo Gym is in play; teacher scoring silently skips → late `ValueError` (or silent hang on older gym) — candidate `fix:` PR mirroring `assert_prev_logprobs_available`.
3. `docs/about/algorithms/mopd.md` doesn't document the dataset row schema; CI's `nanov3_data` is pre-staged internal cluster state, so the recipe is not reproducible externally as checked in — candidate `docs:` PR (+ optionally upstream `tools/prepare_mopd_gym_data.py`).
4. Minor: `get_teacher_routing_metrics()` in `nemo_rl/algorithms/opd.py` has no call site.

See `mopd_design_analysis.md` for the feature's functional design, data/control-flow
charts, and code map.
