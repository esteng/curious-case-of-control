#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_multi_gpu
#SBATCH -p brtx6
#SBATCH --gpus=2

#python -u run_experiment.py --cfg ${CONFIG}
#deepspeed --num_gpus 2 hf_tools/hf.py 
python -u hf_tools/hf.py
