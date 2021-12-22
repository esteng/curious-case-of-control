#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_gpu
#SBATCH -p brtx6
#SBATCH --gpus=1

python -u run_experiment.py --cfg ${CONFIG}
