# Root cause: reward drop on resume from step 120 (grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-bf16-20260721)

**Date:** 2026-07-21 · **Author:** debugging session (Claude) · **Status:** root cause confirmed

## TL;DR

Every checkpoint the original run (job 2417437) saved after its first few saves contains **stale
weights and optimizer state frozen at the job's first save (step 10)** — bit-identical across step_90,
step_110, and step_120. The resume (job 2417816) faithfully loaded step_120 and therefore
restarted from a ~step-30/40-quality policy: train reward fell from −0.16 to −0.88 and
val accuracy from 0.195 to 0.078 (≈ exp_001's step-30 value 0.070).

The defect is **nvidia-resiliency-ext 0.6.0's async-checkpoint GPU-IPC data cache**
(`use_cached_data_structure`), enabled via the recipe's
`policy.megatron_cfg.checkpoint.ckpt_assume_constant_structure: true` + `async_save: true`.
Training itself was healthy the whole time — only the bytes written to disk were wrong.

## Evidence

| # | Observation | Source |
|---|---|---|
| 1 | Resume job's first validation (step 130, i.e. after 10 re-trained steps) = **0.0781** vs 0.1953 at live step 120; the step-121 rollout fingerprint (reward −0.883, gen len 774) matches exp_001's **step 10** (−0.854, len 787) — the frozen content is the job's *first save* | driver logs 2417437 / 2417816, train_data jsonl |
| 2 | Model tensors (embedding, qkv, proj, output_layer) **bit-identical** across step_90/110/120 (`frac elements differing = 0.000`); step_130 (resume job's first save) fully fresh | `ckpt_forensics_round2.py` |
| 3 | Adam `exp_avg` and main-param buckets **bit-identical** across step_90/110/120 — impossible for a live optimizer | same |
| 4 | `common.pt` per checkpoint is **current** (scheduler `num_steps` = 46080/56320/61440 = step×512) — the synchronous write path is fine; only async-written dcp tensor data is frozen | round-1 forensics |
| 5 | vLLM vs megatron logprobs agree post-resume (mean \|Δlp\| ≈ 0.02) — refit path fine; both engines ran the same (stale) weights | exp_002 train_data jsonl |
| 6 | Optimizer/LR scheduler/dataloader restore worked correctly (lr continuous at 1e-06 across boundary; probs_ratio = 1) | wandb sweep |
| 7 | Config identical between segments (only `log_dir` differs); same container image (v4) both jobs | step_120 vs step_130 config.yaml |

## Mechanism

1. Recipe sets `megatron_cfg.checkpoint: {async_save: true, ckpt_assume_constant_structure: true}`
   (`examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-bf16.yaml`).
2. Megatron-Bridge → Megatron-Core `TorchDistSaveShardedStrategy.async_save(async_strategy="nvrx")`
   (default; mcore's own async path is deprecated —
   `3rdparty/.../Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:658-680`)
   passes `use_cached_data_structure = ckpt_assume_constant_structure` to NVRx's
   `FileSystemWriterAsync` (torch.py:710).
3. **nvidia-resiliency-ext 0.6.0** `checkpointing/async_ckpt/filesystem_async.py`
   `prepare_write_data`, GPU-IPC path:
   - first cached save: `cached_tensor_data = (gpu_items, gpu_data)` → GPU tensors go to the
     persistent worker via **CUDA IPC**; worker stores them in the class-level
     `PersistentAsyncCaller._worker_data_cache` (core.py:434-438).
   - every later save: `cache_exists` → `cached_tensor_data = None` — *"Signal to reuse cached
     data"* — **no tensor data is sent**; the worker re-reads GPU memory through the first
     save's cached IPC handles.
4. That design assumes model/optimizer tensors occupy the same GPU storage for the entire run
   (true in classic megatron pretraining). NeMo-RL **colocated GRPO churns GPU memory every
   step** (model/optimizer offload→reload around vLLM generation, plus per-save temporary
   tensors from state-dict transforms), so the cached IPC handles decouple from the live
   weights. The cache is primed on the job's first save (step 10 here), and every subsequent
   checkpoint re-writes that first save's content.
5. A new job spawns a fresh worker with an empty cache — which is why the resume job's first
   save (step_130) contains real (current) data, and why the original run's own training was
   never affected.

Notably, NVRx 0.6.0 also ships a `use_cpu_shm_for_gpu_tensors` (cpu_shm_mode) path that
re-copies fresh values into shared-memory tensors on **every** save — an implicit
acknowledgment that the GPU-IPC cache is unsafe when storages change. Megatron-Bridge does not
enable that mode.

## Blast radius

- **All checkpoints of exp_001 (step_90/110/120) are stale** — none is a usable resume point.
- step_130 is a real snapshot but of the damaged (post-bad-resume) lineage — discard.
- **NVFP4 run confirmed equally affected** (`ckpt_forensics_nvfp4.py`): step_120 ≡ step_130
  bit-identical (model weights + Adam exp_avg frozen near-base); step_140 (resume job
  2417823's first save) is fresh but belongs to a resumed-from-stale lineage — its post-resume
  segment is damaged the same way the bf16 one was. Nuance: the optimizer main-param buckets
  differ slightly between 120 and 130 — CPU-resident tensors (offloaded optimizer main params)
  are always passed fresh; **only GPU-resident tensors are cached/frozen**, so the corruption
  is non-uniform and even harder to spot.
- **Every NeMo-RL megatron run with `async_save: true` + `ckpt_assume_constant_structure:
  true` is suspect** — and these are the *defaults* inherited from
  `examples/configs/grpo_math_1B.yaml:161-162`, with NVRx v0.6.0 pinned in stock `main`'s
  uv.lock, so this is an upstream NeMo-RL bug, not specific to this env. Anything saved with
  more than one checkpoint per job is at risk; first-save-per-job checkpoints are fine.
- Wandb metrics of exp_001 are valid (training was healthy); only on-disk state is wrong.

## Fix / mitigation

Immediate (recovery run):
- `policy.megatron_cfg.checkpoint.async_save: false` — fully safe, synchronous saves
  (recommended for the re-run), or
- keep `async_save: true` but set `ckpt_assume_constant_structure: false` — NVRx then sends
  fresh GPU tensors via IPC on every save (no cross-save data cache). Plan-cache perf is lost;
  data is correct.

Proper fixes (upstream):
- NeMo-RL: never enable `ckpt_assume_constant_structure` together with colocated
  generation/offload; or force NVRx cpu_shm_mode.
- NVRx: invalidate `_worker_data_cache` when source storages change (or checksum-validate),
  or default to the cpu-shm re-copy path.
- File issues against nvidia-resiliency-ext (v0.6.0 `filesystem_async.py` GPU-IPC reuse path)
  and Megatron-Bridge (exposes the unsafe combination by default in RL settings).

Guardrail:
- Add a save→reload verification to the smoke suite (Tier-A `ckpt-resume`): after the second
  checkpoint of a run, reload it and compare a weight checksum against the live model (or val
  parity ±0.03). This exact failure is invisible without cross-save comparison because each
  save "succeeds" and each load "succeeds".

## Recovery recommendation

1. Re-run the bf16 baseline from scratch with `async_save: false` (fresh EXP_TAG, fresh
   checkpoint dir). exp_001's wandb curve remains the reference.
2. Before the long run, verify the fix: run ~25 steps with save_period 10, then compare
   step_10 vs step_20 checkpoints (they must differ; `cmp` on a .distcp shard or the round-2
   forensics script) and reload step_20 with val_at_start to confirm val parity.
3. Audit NVFP4 runs the same way (pairwise-compare their kept checkpoints) before trusting
   any of their resumed segments.

## Verification artifacts

- Probe jobs (reload each checkpoint fresh; val fires at the next val_period boundary, i.e.
  after ~10 re-trained steps — post-resume validation does not run at start):
  - 2418684 reload **step_110** → val@120' = **0.0859** (live run's step-120 val was 0.1953)
  - 2418685 reload **step_90** → val@100' = **0.0586** (live run's step-100 val was 0.1562)
  - resume 2417816 reload **step_120** → val@130' = **0.0781**
  All three cluster at ≈0.074 ± noise despite loading nominally different checkpoints — the
  same frozen policy plus ~10 steps of retraining, exactly as the bit-identical tensor
  forensics predicted. No exp_001 checkpoint is usable.
- Forensics scripts: `research/vllm-nvfp4-pertoken/debug_resume/ckpt_forensics{,_round2}.py`,
  `wandb_boundary_sweep.py`.
- Checkpoint backups: `results/grpo-...-bf16-20260721-forensics/` (hardlinks).
