#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_cpu 
#SBATCH -p brtx6

python -u ${SCRIPT}
