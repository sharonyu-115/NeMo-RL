#!/bin/bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=4:0:0
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=env-refresh-mopd
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/shuangy/logs/env-refresh-mopd-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/shuangy/logs/env-refresh-mopd-%j.err

# Export secrets on the submit host so they propagate into the container env.
set -a; source ~/.env; set +a

srun \
  --no-container-mount-home \
  --container-image=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/images/nemo-rl-fp4-fa4-venvs-sm90fix-2026-07-10.sqsh \
  --container-mounts=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy,/lustre/fsw/portfolios/coreai/users/shuangy:/lustre/fsw/portfolios/coreai/users/shuangy \
  --container-save=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/images/nemo-rl-mopd-main-2026-07-15.sqsh \
  --container-writable \
  bash /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/src/NeMo-RL/nemo-rl-mopd/env_refresh_mopd.sh
