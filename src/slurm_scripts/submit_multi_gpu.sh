#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_multi_gpu4
#SBATCH -p brtx6
#SBATCH --gpus=2

echo "Visible: ${CUDA_VISIBLE_DEVICES}" 
python -u run_experiment.py --cfg ${CONFIG}
