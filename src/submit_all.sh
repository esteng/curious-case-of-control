#!/bin/bash

for config in gpt_neo_2.7b_subject_control.yaml gpt_neo_2.7b_object_control_passive.yaml gpt_j_object_control.yaml gpt_j_subject_control.yaml gpt_j_object_control_passive.yaml t0_object_control.yaml t0_subject_control.yaml t0_object_control_passive.yaml
do 
    path="slurm_scripts/configs_just_prompt_agent/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch slurm_scripts/submit_gpu.sh --export 
    sleep 300;
done 
