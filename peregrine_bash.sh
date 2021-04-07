#!/bin/bash

#SBATCH --time=04:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load libpciaccess/0.16-GCCcore-9.3.0
module load Python/3.8.6-GCCcore-10.2.0

# source /data/s3219496/.envs/gan_env/bin/activate
source /data/s2967383/.envs/gan_env/bin/activate

#python3 launch.py test VANGAN
#python3 launch.py test default
python3 launch.py paintings VANGAN
#python3 launch.py paintings default
#python3 launch.py paris VANGAN
#python3 launch.py paris default
