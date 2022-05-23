# The Curious Case of Control 

Code and prompts for the paper "The Curious Case of Control" 

## Code 
Prompts were run through APIs (GPT3, Jurassic) as well as through huggingface. 

API calls were made in .ipynb files, located in the respective directories for the experiment type (e.g. `subject_control`)  
Huggingface models were run locally using bash scripts and yaml config files (see below). 

Experiments for HF models were run using `src/run_experiment.py`. The input to that script is a .yaml config file specifiying commandline arguments. 
OpenAI and Jurassic require api keys. These are expected to be stored in `src/oai_api.key` and `src/jurassic_api.key` 

### Huggingface experiment configs 
The submission scripts are in the `src` directory. Each script loops over the relevant configs, located in `src/hf_configs/configs`.
These `.yaml` config files specify the commandline arguments to the `src/run_experiment.py` script.  

### Prompts 
The prompts were generated dynamically from the templates in `src/api_tools.py`.   


## Results 
Raw results from each prompt type are stored in the directories for the experiment that do not have the `to_plot` suffix. 
These results are post-processed in `.ipynb` files and then stored in a directory with the same name, but with a `to_plot` suffix. 
In cases where multiple templates are compared, this comparison is done using the results from the `to_plot` directories. 

Results where the prompt has instructions are contained in the `with_instructions` dir. Results for prompts with abridged instructions are in `short_instructions`. 
The max over instruction types is found in `best_instructions`.  


### Visualization 
Visualizations are in `src/vis.ipynb` with functions in `src/vis.py`. All statistical significance tests are done in `src/comapare.ipynb`. 

### Data statistics 
In the paper, we analyze the frequency of subject/object control constructions in a sample of C4. This sample is too large for github, and is available [here](https://nlp.jhu.edu/control_data/sample.json). 

