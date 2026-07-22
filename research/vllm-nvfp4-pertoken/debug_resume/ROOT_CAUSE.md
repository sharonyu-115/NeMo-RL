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

## Mechanism diagrams

### Save flow under async_save (per checkpoint step)

```
 GRPO step-N save (every save_period steps)                 [TRAINER PROCESS]
 ────────────────────────────────────────────────────────────────────────────
  grpo.py:3125-3158
  ┌──────────────────────────────────────────────┐
  │ init_tmp_checkpoint(tmp_step_N/)             │  writes training_info.json,
  │                                              │  config.yaml  (plain files — always fresh)
  └──────────────┬───────────────────────────────┘
                 ▼
  megatron_policy_worker.py:2333  save_checkpoint()
  ┌──────────────────────────────────────────────┐
  │ 1. finalize any previous async save (block)  │
  │ 2. cfg.checkpoint.save = tmp_step_N/weights  │
  │ 3. disable forward pre-hook (param gather)   │
  └──────────────┬───────────────────────────────┘
                 ▼
  megatron-bridge checkpointing.py:1018  save_checkpoint()
  ┌──────────────────────────────────────────────┐
  │ generate_state_dict(model, optimizer, …)     │  fresh sharded refs to LIVE
  │                                              │  GPU tensors — correct so far
  │ common.pt  ◄── written SYNCHRONOUSLY, rank 0 │  (why scheduler counters were
  │                                              │   always current in bad ckpts)
  └──────────────┬───────────────────────────────┘
                 ▼
  mcore torch.py:658  TorchDistSaveShardedStrategy.async_save(async_strategy="nvrx")
  ┌──────────────────────────────────────────────┐
  │ nvidia-resiliency-ext installed? ──── yes ──►│  NVRx writer takes over
  │ (mcore's own async path is deprecated)       │  use_cached_data_structure =
  │                                              │    ckpt_assume_constant_structure ◄ recipe knob
  └──────────────┬───────────────────────────────┘
                 ▼
  NVRx filesystem_async.py  prepare_write_data()
  ┌──────────────────────────────────────────────┐
  │ resolve plan items → live GPU tensors        │
  │ split: GPU tensors  vs  CPU/ByteIO tensors   │
  │        │                       │             │
  │        │                       └────────────►│  CPU/ByteIO: ALWAYS sent fresh
  │        ▼                                     │  (offloaded optim main params
  │  use_cached_data_structure?                  │   stayed current in bad ckpts)
  │        │yes                                  │
  │        ▼                                     │
  │  ╔═══ cache_exists for this key? ══════════╗ │
  │  ║ NO  (first save of the job)             ║ │
  │  ║  cached_tensor_data = (items, tensors)  ║ │  GPU tensors pickled via
  │  ║  → send to worker via CUDA IPC          ║ │  CUDA IPC handles
  │  ║  → worker caches handles in             ║ │
  │  ║    _worker_data_cache   (core.py:434)   ║ │
  │  ║                                         ║ │
  │  ║ YES (every later save)           ⚠ BUG  ║ │
  │  ║  cached_tensor_data = None              ║ │  ◄── "signal to reuse cached
  │  ║  → NO tensor data sent at all           ║ │       data" (fs_async.py:405-421)
  │  ╚═════════════════╤═══════════════════════╝ │
  └────────────────────┼─────────────────────────┘
                       ▼
 ────────────────────────────────────────────────────────────────────────────
  NVRx PersistentAsyncCaller.async_loop            [PERSISTENT WORKER PROCESS,
  (spawned once per job, lives across saves)        spawned on the FIRST save]
  ┌──────────────────────────────────────────────┐
  │ preload_fn(): D2H copy of GPU tensors        │
  │   first save : reads tensors just received   │  ✓ correct bytes
  │   later saves: reads GPU memory through the  │  ⚠ STALE — colocated GRPO
  │     IPC handles cached on the FIRST save     │    offloads/reallocates params
  │                                              │    every step, so these handles
  │ write .distcp shards → tmp_step_N/weights    │    no longer point at the live
  │ signal completion                            │    weights. Every later ckpt
  └──────────────┬───────────────────────────────┘    re-writes first-save state.
                 ▼
 ────────────────────────────────────────────────────────────────────────────
  back in trainer: grpo.py                          [TRAINER PROCESS]
  ┌──────────────────────────────────────────────┐
  │ finalize_async_save(blocking) → .metadata,   │
  │ latest_checkpointed_iteration.txt            │
  │ rename tmp_step_N/ → step_N/    "success" ✓  │  ← looks perfect from outside
  └──────────────────────────────────────────────┘
```

The two escape hatches map onto the fork: `async_save: false` skips the entire NVRx
half (sync writer resolves and writes live tensors in-process);
`ckpt_assume_constant_structure: false` keeps async but takes the "NO" branch every
save (fresh IPC tensors each time).

### Pointer-level view: why later saves read frozen memory

An IPC handle names a physical **allocation**, not a tensor. Three snapshots of one
GPU shared by the trainer and the persistent checkpoint worker:

**T1 — first save (step 10). Cache gets primed.**

```
TRAINER PROCESS                      GPU PHYSICAL MEMORY              CKPT WORKER PROCESS
───────────────                      ───────────────────              ───────────────────
model.param.data ─────────────┐      ┌─────────────────────┐
(torch.Tensor)                └────► │ Allocation A        │ ◄──┐     _worker_data_cache[key]:
                                     │ bytes = W@step10    │    │       [(item, mapped_tensor)]
optimizer.exp_avg ──────────► [ .. ] │ (param buffer,      │    └───── mapped storage M_A
                                     │  cudaMalloc'd once  │           (cudaIpcOpenMemHandle(A))
save #1: pickle tensors  ──────────► │  by DDP at startup) │
  = cudaIpcGetMemHandle(A)           └─────────────────────┘     D2H copy reads A → W@step10 ✓
  handle crosses process boundary,                                writes correct step_10 ckpt
  NOT the bytes                       shared-mem refcount file:
                                      counter[A] = 1  (worker holds a mapping)
```

**T2 — colocated offload/onload (every step, between saves).**

```
TRAINER PROCESS                      GPU PHYSICAL MEMORY              CKPT WORKER PROCESS
───────────────                      ───────────────────              ───────────────────
offload: param.data → CPU;           ┌─────────────────────┐
GPU tensor A "freed"                 │ Allocation A        │ ◄────── M_A still mapped!
  └► allocator checks counter[A]=1   │ bytes = W@step10    │         (cache never released
     → CANNOT reuse/free A           │ FROZEN — nobody     │          until worker shutdown)
     → A parked in CudaIPCSentData   │ writes here again   │
       **Limbo** (pinned forever)    ├─────────────────────┤
                                     │ Allocation B (NEW)  │
onload: param.data ────────────────► │ bytes = W@step11,   │         worker knows nothing
(fresh cudaMalloc → different        │ 12, 13… updated     │         about B — no handle
 allocation, different address)      │ in place by training│         was ever sent for it
                                     └─────────────────────┘
```

**T3 — save #2 (step 20). The bug fires.**

```
TRAINER PROCESS                      GPU PHYSICAL MEMORY              CKPT WORKER PROCESS
───────────────                      ───────────────────              ───────────────────
generate_state_dict() resolves       ┌─────────────────────┐
LIVE tensors → point into B ✓        │ A: W@step10 (stale, │ ◄────── cache_exists=True, so
                                     │    pinned by limbo) │         trainer sent NOTHING;
NVRx prepare_write_data:             ├─────────────────────┤         worker D2H-copies from
  cache_exists(key) → True           │ B: W@step20 (live)  │         its cached M_A → reads
  cached_tensor_data = None ────X    └─────────────────────┘         **W@step10**
  (no handles for B ever sent)
                                                                     step_20 ckpt on disk =
                                                                     W@step10 bytes  ✗✗✗
```

Key pointer facts:

1. **An IPC handle names an allocation, not a tensor.** Tensors are (allocation,
   offset, shape) views; when the trainer's params move to allocation B, the
   worker's ticket to A doesn't follow and nobody invalidates it.
2. **CUDA IPC refcounting makes the staleness *silent*.** PyTorch tracks each
   exported allocation with a shared-memory refcount (`CudaIPCSentData` /
   `CudaIPCSentDataLimbo`, torch/csrc/CudaIPCTypes.h). Because the worker still
   holds a mapping, the trainer-side free at T2 cannot release A — it is parked in
   limbo, pinned and never rewritten. The worker's reads therefore never crash and
   never see garbage: A is a perfectly preserved copy of step-10 state. (It is also
   a slow GPU memory leak — limbo blocks are held for the whole job.)
3. **Classic pretraining never hits T2** — there, allocation A *is* the permanent
   DDP param buffer and optimizer steps write into it in place forever, so the
   cached mapping is always current and the cache is a legitimate optimization.
   Colocated RL's per-step offload/onload breaks the "A is forever" premise, and
   the design has no invalidation hook (`_worker_data_cache` clears only at worker
   shutdown — exactly why each job's *first* save is correct). NVRx itself guards
   the one fresh-allocation case it knew about (dequantized tensors,
   `filesystem_async.py:289-291`) but not framework-level reallocation.

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
