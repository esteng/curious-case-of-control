#!/bin/bash

#for config in gpt_neo_1.3b_subject_control.yaml gpt_j_subject_control.yaml 
for config in gpt_neo_1.3b_subject_control.yaml 
do 
    path="slurm_scripts/configs/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch slurm_scripts/submit_multi_gpu.sh --export;
done 
