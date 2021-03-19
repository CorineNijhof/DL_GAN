#!/bin/bash

#SBATCH --time=02:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python matplotlib
module load libpciaccess/0.16-GCCcore-9.3.0

python3 launch.py
