#!/bin/bash

for config in gpt_j_object_control_passive.yaml gpt_neo_1.3b_object_control_passive.yaml gpt_neo_2.7b_object_control_passive.yaml t0_object_control_passive.yaml 
do 
    path="slurm_scripts/configs_profession/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch  slurm_scripts/submit_cpu.sh --export 
    # sleep 3 hours so jobs don't overlap, causes race condition 
    sleep 10800
done 
