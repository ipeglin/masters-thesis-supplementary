#!/bin/bash
#SBATCH --job-name="run_pipeline"
#SBATCH --account=share-ie-idi
#SBATCH --mail-user=$USER@stud.ntnu.no  # Fixed by injection script
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# Use %u (native Slurm user pattern) for more robustness in paths
#SBATCH --output=/cluster/home/%u/slurm/output_logs/job.%j-%x.out
#SBATCH --error=/cluster/home/%u/slurm/output_logs/job.%j-%x.err

# Variables to be injected by fetch script
PROJECT_ROOT = ""
USER_NAME = ""

# Now use the injected USER_NAME safely in shell commands
mkdir -p /cluster/home/$USER_NAME/slurm/output_logs

PROJECT_ROOT = ""

# Unload all modules
module purge

# Activate environment
bash $PROJECT_ROOT/scripts/init.sh # build hdf5 crate locally (manual clone in ~/Downloads)

source $PROJECT_ROOT/scripts/sys-idun_env.sh

# Run Rust scripts (most steps are almost instantaneous)
bash $PROJECT_ROOT/scripts/run_pipeline.sh
