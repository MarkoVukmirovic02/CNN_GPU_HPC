#!/bin/bash
#SBATCH --job-name=cats-dogs
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --error=%j.err
#SBATCH --output=%j.out

source .venv/bin/activate

which python
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('torchvision:', torchvision.__version__)"

srun python cats_vs_dogs.py
