#!/bin/bash

for config in gpt_j_subject_control.yaml t0_subject_control.yaml gpt_neo_1.3b_subject_control.yaml gpt_neo_2.7b_subject_control.yaml 
do 
    path="slurm_scripts/configs_profession/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch --nodelist=brtx604 slurm_scripts/submit_cpu.sh --export 
    # sleep 3 hours so jobs don't overlap, causes race condition 
    sleep 10800
done 
