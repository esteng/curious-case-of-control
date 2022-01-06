#!/bin/bash

for config in gpt_j_object_control.yaml t0_object_control.yaml 
do 
    path="slurm_scripts/configs_just_agent/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch  slurm_scripts/submit_multi.sh --export 
done 
