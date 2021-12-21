#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/log.out
#SBATCH -p brtx6
#SBATCH --gpus=1

conda activate openai 
python reserve.py 
