#!/usr/bin/env bash
#SBATCH --job-name=linear-regression
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=Linear/slurm-%j.out
#SBATCH --error=Linear/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Uncomment or adapt this line if your cluster requires modules.
# module load julia

export JULIA_NUM_THREADS=1
export SLURM_NTASKS="${SLURM_NTASKS:-${SLURM_NPROCS:-3}}"

julia --project=. Linear/Linear_Regression_HPC.jl
