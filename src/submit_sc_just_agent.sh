#!/bin/bash

for config in  t0_subject_control.yaml  gpt_j_subject_control.yaml
do 
    path="slurm_scripts/configs_just_prompt_agent/${config}"
    export CONFIG=${path}
    echo ${CONFIG}
    sbatch slurm_scripts/submit_multi_gpu.sh --export;
done 
