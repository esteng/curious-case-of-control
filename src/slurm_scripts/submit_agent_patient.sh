#!/bin/bash

model=$1
num=$2


for config in slurm_scripts/agent_patient_${num}/${model}/*.yaml
do
    echo $config
    export CONFIG=$config
    sbatch slurm_scripts/submit_ap_1.sh --export 
done
