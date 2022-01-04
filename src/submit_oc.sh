#!/bin/bash

for config in gpt_neo_2.7b_object_control.yaml gpt_j_object_control.yaml t0_object_control.yaml 
do 
    path="slurm_scripts/configs_profession/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch --nodelist=brtx602 slurm_scripts/submit_cpu.sh --export 
    # sleep 3 hours so jobs don't overlap, causes race condition 
    sleep 10800
done 
