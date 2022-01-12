#!/bin/bash

model=$1


for config in slurm_scripts/agent_patient_2/${model}/*.yaml
do
    echo $config
    export CONFIG=$config
    sbatch slurm_scripts/submit_ap.sh --export 
done
