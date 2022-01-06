#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_multi_gpu
#SBATCH -p brtx6
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
echo "visible:" 
echo $CUDA_VISIBLE_DEVICES
#python -u run_experiment.py --cfg ${CONFIG}
