#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --job-name=samudrace
#SBATCH --output=logs/samudrace-%A.txt

source ~/.bashrc
conda activate samudrace

cd /home/a/antonio/repos/samudrace

/home/a/antonio/miniforge3/envs/samudrace/bin/python3 -m fme.coupled.inference /home/a/antonio/repos/ace/configs/samudrace_config.yaml
