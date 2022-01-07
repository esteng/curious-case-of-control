#!/bin/bash

for config in  gpt_neo_2.7b_subject_control.yaml gpt_j_subject_control.yaml 
do 
    path="slurm_scripts/configs/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch --nodelist=brtx605 slurm_scripts/submit_multi_gpu.sh --export;
    sleep 2000;
done 
