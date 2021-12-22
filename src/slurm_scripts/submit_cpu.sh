#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_cpu 
#SBATCH -p brtx6
#SBATCH --mem=40G   

python -u run_experiment.py --cfg ${CONFIG}
