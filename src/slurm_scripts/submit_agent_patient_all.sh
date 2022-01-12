#!/bin/bash

for num in 1 2
do
    for model in gpt_j gpt_neo_1.3b gpt_neo_2.7b t0 
    do 
        ./slurm_scripts/submit_agent_patient.sh ${model} ${num} 
    done
done
