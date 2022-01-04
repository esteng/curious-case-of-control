#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_cpu3
#SBATCH -p brtx6

python -u run_experiment.py --cfg ${CONFIG}
