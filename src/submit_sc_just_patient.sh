#!/bin/bash

for config in  gpt_neo_2.7b_subject_control.yaml t0_subject_control.yaml 
do 
    path="slurm_scripts/configs_just_prompt_patient/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch --nodelist=brtx602 slurm_scripts/submit_multi_gpu.sh --export;
    sleep 2000;
done 
