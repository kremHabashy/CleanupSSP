#!/bin/bash
#SBATCH --job-name=cleanup_project
#SBATCH --output=logs/out/cleanup_output.log
#SBATCH --error=logs/err/cleanup_error.log
#SBATCH --time=9:00:00  
   
# SBATCH --gres=gpu:1
#SBATCH --gres=gpu:celiasmigpu:1
#SBATCH --partition=CELIASMI

#SBATCH --mem=16G
#SBATCH --cpus-per-task=16


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_INTEROP_THREADS=$SLURM_CPUS_PER_TASK


# Activate virtual environment
source /u1/khabashy/CleanupSSP/CleanUp/bin/activate


# Run your script
python -m experiments.dim_sweep
# python -m cleanup_ssps.main
