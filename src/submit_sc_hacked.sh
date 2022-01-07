#!/bin/bash

#for config in  gpt_neo_1.3b_subject_control.yaml gpt_neo_2.7b_subject_control.yaml t0_subject_control.yaml gpt_j_subject_control.yaml 
for config in gpt_j_subject_control.yaml t0_subject_control.yaml gpt_neo_1.3b_subject_control.yaml gpt_neo_2.7b_subject_control.yaml 
do 
    path="slurm_scripts/configs_hacked/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch slurm_scripts/submit_gpu.sh --export;
done 
