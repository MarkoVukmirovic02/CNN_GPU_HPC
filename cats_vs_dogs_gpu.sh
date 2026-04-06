#!/bin/bash
#SBATCH --job-name=cats-dogs
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --error=%j.err
#SBATCH --output=%j.out

source .venv/bin/activate

srun python cats_vs_dogs.py
