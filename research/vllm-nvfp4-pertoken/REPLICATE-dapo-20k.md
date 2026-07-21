# Replicate the DAPO-20k runs (Qwen3-30B-A3B-Base, GB200)

BF16 baseline + NVFP4 per-token W4A4 rollout, DAPO-512, 20k response,
8 nodes x 4 GPU, CUDA graphs. Same cluster (ptyche/GB200), different path.

## Repo / branch

```
git@github.com:sharonyu-115/RL.git    branch: shuangy/nvfp4-pertoken-probe
```

## Files (all in-repo)

- Launcher: `research/vllm-nvfp4-pertoken/run_dapo_longrun.sh`
- BF16 recipe: `examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-bf16.yaml`
- NVFP4 recipe: `examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken.yaml`
  (inherits the BF16 recipe + `examples/configs/te_precision/attn_bf16_mlp_nvfp4.yaml`)

## Shared assets (read from these; same-cluster lustre)

- Container (v4, TE 937c4de w/ NVTE_NVFP4_ROW_SCALED_ACTIVATION):
  `/lustre/fsw/general_sa/shuangy/images/nemo-rl-nvfp4-pertoken-venvs-gb200-2026-07-20-v4.sqsh`
- Model: `/lustre/fsw/general_sa/shuangy/models/Qwen/Qwen3-30B-A3B-Base`

## `.env` (at `research/vllm-nvfp4-pertoken/.env`, gitignored)

```
RL_DIR=<your clone path>          # CKPT/log dirs auto-derive from this
CONTAINER_IMAGE=<v4 sqsh above>
HF_HOME=<your HF cache>
WANDB_API_KEY=<yours>
WANDB_PROJECT=<yours>
```

## Change for a different path (3 hardcoded to shuangy in the BF16 recipe)

- `policy.model_name` + `policy.tokenizer.name` -> model path
  (shuangy's model dir is share-readable, or point to your own copy)
- `policy.megatron_cfg.env_vars.NRL_MEGATRON_CHECKPOINT_DIR` -> **your own writable dir**

(`checkpointing.checkpoint_dir` / `logger.log_dir` are auto-set by the launcher
from `RL_DIR` + run name, so no need to edit those.)

## Launch (8 nodes x 4 GPU, CUDA graphs on; EXP_TAG = fresh wandb name)

```bash
# BF16
sbatch --export=ALL,GPUS_PER_NODE=4,PRECISION=bf16,EXP_TAG=<tag>,\
CONTAINER_IMAGE_OVERRIDE=<v4 sqsh> research/vllm-nvfp4-pertoken/run_dapo_longrun.sh

# NVFP4 per-token
sbatch --export=ALL,GPUS_PER_NODE=4,PRECISION=nvfp4,EXP_TAG=<tag>,\
CONTAINER_IMAGE_OVERRIDE=<v4 sqsh> research/vllm-nvfp4-pertoken/run_dapo_longrun.sh
```

## Notes

- Account / partitions / node count are in the launcher `#SBATCH` header
  (`general_sa`; `batch,tcpo,36x2-a01r,a02grace`; `--nodes=8`; 5h wall).
- **Resume** (across the 5h wall): re-run the **same command** — it auto-resumes
  from the latest checkpoint and rejoins the pinned W&B run
  (`CKPT_DIR/.wandb_run_id`). ~120s/step, ~5-6 allocations to 800 steps.
- Launcher knobs: `PRECISION={bf16,nvfp4}`, `EXP_TAG` (run-name suffix),
  `MAX_STEPS`, `RECIPE_OVERRIDE` (arbitrary recipe), `CONTAINER_IMAGE_OVERRIDE`.
- Key config facts (already baked into the recipes): vLLM `tp1`; BF16 forces
  `moe_backend=triton` (launcher-injected; the GB200 FlashInfer BF16 MoE bug),
  NVFP4 uses auto (flashinfer_trtllm for its quant experts);
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.8`
  (cudaIpc refit + fragmentation); `gpu_memory_utilization: 0.5`;
  `enforce_eager: false` (CUDA graphs, ~10x faster rollout).
