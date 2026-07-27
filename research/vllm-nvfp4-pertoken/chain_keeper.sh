#!/bin/bash
# Auto-keep the fp4fwd (leg2) + fp4bwd-r3 afterany chains fed until each reaches
# TARGET steps, then cancel leftover queued jobs. Detached (setsid) — survives the
# agent session. Safety: per-chain MAXTOTAL top-up cap + stuck detection.
set -u
RL=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-te-bwd
cd "$RL"
LAUNCH=research/vllm-nvfp4-pertoken/run_dapo_longrun.sh
LOG=session/te-nvfp4-backward/chain_keeper.log
CAP=session/te-nvfp4-backward/.keeper_submitted   # one line per keeper-submitted job

KEYS=(fp4fwd-leg2 fp4bwd-r3 fp4fwd-rhtsr fp4bwd-r3-rhtsr)
declare -A RECIPE=(
  [fp4fwd-leg2]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4fwd.yaml
  [fp4bwd-r3]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3.yaml
  [fp4fwd-rhtsr]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4fwd-rhtsr.yaml
  [fp4bwd-r3-rhtsr]=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3-rhtsr.yaml )
declare -A RUNDIR=(
  [fp4fwd-leg2]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4fwd
  [fp4bwd-r3]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3
  [fp4fwd-rhtsr]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4fwd-rhtsr
  [fp4bwd-r3-rhtsr]=results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4bwd-r3-rhtsr )
declare -A JOBNAME=(
  [fp4fwd-leg2]=general_sa-nemo_rl.fp4fwd-leg2-1500
  [fp4bwd-r3]=general_sa-nemo_rl.fp4bwd-r3-1500
  [fp4fwd-rhtsr]=general_sa-nemo_rl.fp4fwd-rhtsr-1500
  [fp4bwd-r3-rhtsr]=general_sa-nemo_rl.fp4bwd-r3-rhtsr-1500 )

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

    if [ "$n" -le "$LOWWATER" ] && [ -n "$last" ]; then
      submitted=$(wc -l < "$CAP" 2>/dev/null || echo 0)
      chain_submitted=$(grep -c "^$key " "$CAP" 2>/dev/null || echo 0)
      if [ "$chain_submitted" -ge "$MAXTOTAL" ]; then log "$key at MAXTOTAL($MAXTOTAL) cap — not topping"; continue; fi
      export RECIPE_OVERRIDE=${RECIPE[$key]}
      prev=$last; added=0
      for i in $(seq 1 $TOPUP); do
        ji=$(sbatch --parsable --dependency=afterany:$prev --job-name="$name" --nodes=8 --time=05:00:00 \
             --partition=batch,tcpo,36x2-a01r,a02grace --export=ALL \
             --output=logs/${key}-1500-%j.out --error=logs/${key}-1500-%j.err "$LAUNCH" 2>/dev/null)
        [ -n "$ji" ] || { log "$key sbatch failed at i=$i"; break; }
        echo "$key $ji" >> "$CAP"; prev=$ji; added=$((added+1))
      done
      log "$key step=$step queued=$n LOW -> added $added afterany jobs (tail $last)"
    else
      log "$key step=$step queued=$n ok"
    fi
  done
  [ "$alldone" = 1 ] && { log "ALL runs reached $TARGET — keeper exiting"; break; }
  sleep "$INTERVAL"
done
