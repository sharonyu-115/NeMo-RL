#!/bin/bash
# Auto-keep the NVFP4 weight-geometry afterany chains (w1d / w2d / w2d-r3) fed until
# each reaches TARGET steps, then cancel leftover queued jobs. Detached (setsid) —
# survives the agent session. Safety: per-chain MAXTOTAL top-up cap, stuck detection,
# and a mandatory per-leg container pin (see the CONTAINER map).
set -u
RL=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd
cd "$RL"
LAUNCH=research/vllm-nvfp4-pertoken/run_dapo_longrun.sh
LOG=session/te-nvfp4-backward/chain_keeper.log
CAP=session/te-nvfp4-backward/.keeper_submitted   # one line per keeper-submitted job

# fp4fwd-rhtsr / fp4bwd-r3-rhtsr REMOVED 2026-07-26: gen_kl exploded to ~3.8
# (vs ~0.005 normal) — forward RHT mismatches the no-RHT vLLM rollout. Cancelled.
# fp4bwd-r3 REMOVED 2026-07-28: stopped at step 764 (enough steps); leaving the key
# in would silently resurrect the leg on the next top-up.
# fp4fwd-leg2 REMOVED 2026-07-28: stalled at step 467 for ~20h, undiagnosed. Re-add
# only after someone looks at why it stopped advancing.
# fp4bwd-w1d / fp4bwd-w2d REMOVED 2026-07-28: both DIVERGED (w1d ckpt step_320,
# w2d ckpt step_360) and were cancelled. Their map entries below are kept so the
# legs can be re-enabled by name, but they must stay OUT of KEYS — the drained-chain
# restart added on 2026-07-28 would otherwise resubmit a fresh head for each.
KEYS=(fp4bwd-w1d-r3 fp4bwd-w2d-r3)
declare -A RECIPE=(
  [fp4bwd-w1d]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w1d.yaml
  [fp4bwd-w1d-r3]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w1d-r3.yaml
  [fp4bwd-w2d]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w2d.yaml
  [fp4bwd-w2d-r3]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w2d-r3.yaml )
declare -A RUNDIR=(
  [fp4bwd-w1d]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w1d
  [fp4bwd-w1d-r3]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w1d-r3
  [fp4bwd-w2d]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w2d
  [fp4bwd-w2d-r3]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-w2d-r3 )
declare -A JOBNAME=(
  [fp4bwd-w1d]=general_sa-nemo_rl.fp4bwd-w1d-1500
  [fp4bwd-w1d-r3]=general_sa-nemo_rl.fp4bwd-w1d-r3-1500
  [fp4bwd-w2d]=general_sa-nemo_rl.fp4bwd-w2d-1500
  [fp4bwd-w2d-r3]=general_sa-nemo_rl.fp4bwd-w2d-r3-1500 )
# MANDATORY per leg. run_dapo_longrun.sh only guards that the image name matches
# *te690*, and .env's default (nemo-rl-te690ffea-probe.sqsh, TE 690ffea4) matches
# that pattern while LACKING the NVTE_NVFP4_PER_TOKEN_WEIGHT_PER_TENSOR_1D patch —
# so an unset override would silently downgrade fp4bwd-w1d from leg C to leg B
# mid-chain and corrupt the run. Keep every leg pinned to the image it started on.
declare -A CONTAINER=(
  [fp4bwd-w1d]=/lustre/fsw/general_sa/shuangy/images/nemo-rl-te690-w1d-probe.sqsh
  [fp4bwd-w1d-r3]=/lustre/fsw/general_sa/shuangy/images/nemo-rl-te690-w1d-probe.sqsh
  [fp4bwd-w2d]=/lustre/fsw/general_sa/shuangy/images/nemo-rl-te690-w1d-probe.sqsh
  [fp4bwd-w2d-r3]=/lustre/fsw/general_sa/shuangy/images/nemo-rl-te690-w1d-probe.sqsh )

TARGET=1500; LOWWATER=3; TOPUP=6; MAXTOTAL=30; INTERVAL=1800; STUCK_CYCLES=4
export PRECISION=nvfp4_bwd GPUS_PER_NODE=4
declare -A LASTSTEP; declare -A STALL
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
log "keeper started (target=$TARGET lowwater=$LOWWATER topup=$TOPUP interval=${INTERVAL}s)"

while true; do
  alldone=1
  for key in "${KEYS[@]}"; do
    dir=${RUNDIR[$key]}; name=${JOBNAME[$key]}
    step=$(ls -d "$dir"/step_* 2>/dev/null | sed 's#.*step_##' | sort -n | tail -1); step=${step:-0}
    n=$(squeue -u "$USER" -h -n "$name" -o "%i" 2>/dev/null | wc -l)
    last=$(squeue -u "$USER" -h -n "$name" -o "%i" 2>/dev/null | sort -n | tail -1)

    if [ "$step" -ge "$TARGET" ]; then
      # done: cancel any leftover QUEUED (not running) jobs to free the queue
      leftovers=$(squeue -u "$USER" -h -n "$name" -t PENDING -o "%i" 2>/dev/null | tr '\n' ' ')
      [ -n "$leftovers" ] && { scancel $leftovers 2>/dev/null; log "$key DONE step=$step — cancelled leftovers: $leftovers"; } || log "$key DONE step=$step"
      continue
    fi
    alldone=0

    # stuck detection: step not advancing across cycles
    if [ "${LASTSTEP[$key]:-(-1)}" = "$step" ]; then STALL[$key]=$(( ${STALL[$key]:-0} + 1 )); else STALL[$key]=0; fi
    LASTSTEP[$key]=$step
    if [ "${STALL[$key]}" -ge "$STUCK_CYCLES" ]; then
      log "$key STUCK: step=$step unchanged for ${STALL[$key]} cycles (jobs=$n) — NOT topping up; needs a look"
      continue
    fi

    if [ "$n" -le "$LOWWATER" ]; then
      chain_submitted=$(grep -c "^$key " "$CAP" 2>/dev/null || echo 0)
      if [ "$chain_submitted" -ge "$MAXTOTAL" ]; then log "$key at MAXTOTAL($MAXTOTAL) cap — not topping"; continue; fi
      export RECIPE_OVERRIDE=${RECIPE[$key]}
      # Pin the image per leg — see the CONTAINER map comment. Never let this be unset.
      export CONTAINER_IMAGE_OVERRIDE=${CONTAINER[$key]}
      # n==0 means the whole chain drained (all jobs finished, or all died as in the
      # w1d wandb-init failures). Restart with a fresh dependency-free head instead of
      # leaving the leg stranded; STUCK detection + MAXTOTAL bound the damage if the
      # leg is genuinely broken.
      if [ -z "$last" ]; then
        log "$key chain DRAINED at step=$step — restarting with a fresh head"
      fi
      prev=$last; added=0
      for i in $(seq 1 $TOPUP); do
        DEP=""; [ -n "$prev" ] && DEP="--dependency=afterany:$prev"
        ji=$(sbatch --parsable $DEP --job-name="$name" --nodes=8 --time=05:00:00 \
             --partition=batch,tcpo,36x2-a01r,a02grace --export=ALL \
             --output=logs/${key}-1500-%j.out --error=logs/${key}-1500-%j.err "$LAUNCH" 2>/dev/null)
        [ -n "$ji" ] || { log "$key sbatch failed at i=$i"; break; }
        echo "$key $ji" >> "$CAP"; prev=$ji; added=$((added+1))
      done
      log "$key step=$step queued=$n LOW -> added $added afterany jobs (image=$(basename ${CONTAINER[$key]}) tail=${last:-none})"
    else
      log "$key step=$step queued=$n ok"
    fi
  done
  [ "$alldone" = 1 ] && { log "ALL runs reached $TARGET — keeper exiting"; break; }
  sleep "$INTERVAL"
done
