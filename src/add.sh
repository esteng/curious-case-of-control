for file in slurm_scripts/configs*_long/*.yaml
do
    echo ${file} 
    echo "" >> ${file}
    echo "long_instruction: True" >> ${file} 
done
