#!/bin/bash
#SBATCH --job-name=cleanup_project
#SBATCH --output=logs/out/dim_sweep.log
#SBATCH --error=logs/err/dim_sweep.log
#SBATCH --time=1:30:00  
   
# SBATCH --gres=gpu:0
#SBATCH --gres=gpu:celiasmigpu:1
#SBATCH --partition=CELIASMI

#SBATCH --mem=16G
#SBATCH --cpus-per-task=32


# Activate virtual environment
source /u1/khabashy/CleanupSSP/CleanUp/bin/activate

# python -m experiments.dim_sweep
# python -m experiments.ls_sweep
# python -m experiments.steps_evaluation_RF
# python -m experiments.avg_perf_vs_dim
# python -m experiments.performance_comparisons
python -m experiments.baselines_compare
# python -m experiments.compute_estimate
# python -m experiments.OT_strength_exploration
# python -m experiments.train_ff_batch_size_sweep