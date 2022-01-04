#!/bin/bash

#for config in gpt_j_object_control.yaml gpt_j_subject_control.yaml 
for config in t0_object_control.yaml t0_subject_control.yaml t0_object_control_passive.yaml
do 
    path="slurm_scripts/configs_profession/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch --nodelist=brtx605 slurm_scripts/submit_cpu.sh --export 
    # sleep 3 hours so jobs don't overlap, causes race condition 
    sleep 10800
done 
