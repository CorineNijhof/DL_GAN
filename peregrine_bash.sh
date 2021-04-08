#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module load libpciaccess/0.16-GCCcore-9.3.0
module load Python/3.8.6-GCCcore-10.2.0

# source /data/s3219496/.envs/gan_env/bin/activate
source /data/s2967383/.envs/gan_env/bin/activate

#python3 launch.py test default
#python3 launch.py test VANGAN
#python3 launch.py data default 0.0002 Adam
#python3 launch.py drawings default 0.0002 Adam
#python3 launch.py paris default 0.0002 Adam
#python3 launch.py paintings default 0.0002 Adam
#python3 launch.py paintings VANGAN 0.0002 Adam
#python3 launch.py data VANGAN 0.0002 Adam 550
#python3 launch.py drawings VANGAN 0.0002 Adam 250
python3 launch.py data default128 0.0002 Adam 450
#python3 launch.py paintings default 0.01 Adam
#python3 launch.py paintings VANGAN 0.01 Adam
#python3 launch.py paintings default 0.0002 SGD
#python3 launch.py paintings VANGAN 0.0002 SGD
#python3 launch.py paintings default 0.01 SGD
#python3 launch.py paintings VANGAN 0.01 SGD
#python3 launch.py paris default
#python3 launch.py paris VANGAN
