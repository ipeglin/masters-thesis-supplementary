#!/bin/bash
#SBATCH --job-name="run-pipeline"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --nodes=2                   # Allocate nodes for the job
#SBATCH --mem=64G
#SBATCH -c28                        # Number of cores (can vary)
#SBATCH --time=03-00:00:00          # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
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

# Unload all modules
module purge

# Activate environment
bash $PROJECT_ROOT/scripts/init.sh # build hdf5 crate locally (manual clone in ~/Downloads)

source $PROJECT_ROOT/scripts/sys-idun_env.sh

# Run Rust scripts (most steps are almost instantaneous)
bash $PROJECT_ROOT/scripts/run_pipeline_with_debug_mode.sh
