#!/bin/bash
#SBATCH --job-name=cleanup_project
#SBATCH --output=logs/out/ls_cleanup_error.log
#SBATCH --error=logs/err/ls_cleanup_error.log
#SBATCH --time=12:00:00  
   
#SBATCH --gres=gpu:0
#SBATCH --gres=gpu:celiasmigpu:0
#SBATCH --partition=CELIASMI

#SBATCH --mem=16G
#SBATCH --cpus-per-task=16


# Activate virtual environment
source /u1/khabashy/CleanupSSP/CleanUp/bin/activate

python -m utils.generate_randomSSPs
