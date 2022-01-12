#!/bin/bash 

#SBATCH -o /home/estengel/child-lm/logs/submit_ap
#SBATCH -p brtx6
#SBATCH --gpus=1

python -u run_agent_patient.py --cfg ${CONFIG}
