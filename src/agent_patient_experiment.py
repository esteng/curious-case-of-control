from typing import Any, Dict
import time 
import pdb 
import json 
import re
import pathlib
import sys 

import pandas as pd 
from tqdm import tqdm 

from api_tools import FixedGPTPrompt, run_experiment
from metrics import StringMetric, AgentPatientStringMetric
from experiment import Experiment

class AgentPatientExperiment(Experiment):
    def __init__(self, 
                model_name: str,    
                exp_type: str,
                prompt_file: str, 
                experiment_fxn: Any, 
                replicants: int, 
                kwargs: Dict):
        super(AgentPatientExperiment, self).__init__(model_name, exp_type, None, experiment_fxn, replicants, kwargs)
        self.prompt_file = prompt_file

    def format_results(self):
        df = pd.DataFrame(columns=['model', 'exp_type', 'prompt', 'pred', 'true', 'response'], dtype=object)
        for line in self.results:
            df = df.append(line, ignore_index=True)
        return df

    def recompute(self): 
        # assert(len(prompt_data) == len(self.results))
        for i, res_line in enumerate(self.results):
            #for line in prompt_data: 
            # line = prompt_data[i]
            # if line['prompt'].strip() == res_line['prompt'].strip():
            # true_ans = line['correct_value']
            true_ans = res_line['true']
            self.results[i]['true'] = true_ans

            response = res_line['response']
            try:
                pred_name = re.sub(r"[()']", "", response).strip()
            except TypeError:
                # print(line)
                pred_name = "other"
            true_name = res_line['true']
            lookup = self.make_lookup() 
            metric = AgentPatientStringMetric(lookup)
            metric(pred_name, prompt=res_line['prompt']) 
            acc, count, __ = metric.get_metric(true_name)
            pred = [k for k,v in count.items() if v > 0][0]
            self.results[i]['pred'] = pred


    def make_lookup(self): 
        lookup = {"Yes": ["Yes", "y", "Y", "yes"], "No": ["no", "No", "n", "N"]}
        return lookup 

    def run(self, 
            overwrite=False, 
            nicknames = None, 
            replicants = 1,
            rate_limit_delay = 60, 
            rate_limit_count=55):
        results_list = []
        num_run_in_time = 0
        start_time = time.time()

        with open(self.prompt_file) as f1:
            prompt_data = json.load(f1)

        lookup = self.make_lookup()
        for prompt_dict in tqdm(prompt_data):
            prompt = prompt_dict['prompt'] 
            true_value = prompt_dict['correct_value']
            metric = AgentPatientStringMetric(lookup) 
            metric, responses = run_experiment(self.experiment_fxn, prompt, replicants, metric, self.kwargs) 
            for resp in responses:
                inner_metric = AgentPatientStringMetric(lookup) 
                inner_metric(resp[1], prompt=prompt)
                acc, count, __ = inner_metric.get_metric(true_value)
                pred = [k for k,v in count.items() if v > 0][0]
                results_line = {"model": self.model_name, "exp_type": self.exp_type,  "true": true_value, "pred": pred, "prompt": prompt, "response": resp[1]}
                self.results.append(results_line)

                num_run_in_time += 1
                end_time = time.time()
                if rate_limit_delay is not None and num_run_in_time > rate_limit_count:
                    # need to pause to not get kicked off 
                    num_seconds_run = end_time - start_time 
                    time_to_sleep = rate_limit_delay - num_seconds_run
                    # add 10 second fudge factor just in case 
                    if time_to_sleep < 0:
                        time_to_sleep = 0 
                    time.sleep(time_to_sleep + 10)
                    # reset 
                    num_run_in_time = 0 
                    start_time = time.time() 




class T5AgentPatientExperiment(AgentPatientExperiment):
    def __init__(self, 
                model_name: str,    
                exp_type: str,
                prompt_file: str, 
                experiment_fxn: Any, 
                replicants: int, 
                kwargs: Dict):
        super(T5AgentPatientExperiment, self).__init__(model_name, exp_type, prompt_file, experiment_fxn, replicants, kwargs)
        self.convert_prompts(prompt_file)

    def convert_prompts(self, prompt_file):
        p = pathlib.Path(prompt_file)

        filename = p.name
        new_filename = f"t5_{filename}"
        new_p = p.parent.joinpath(new_filename)
        with open(p) as f1:
           data= json.load(f1)
        sent_gex = re.compile(r"Sentence: ([^\n]+)")
        quest_gex = re.compile(r"Question: ([^\n]+)")
        new_data =  []
        for pd in data:
            prompt = pd['prompt']
            text = sent_gex.search(prompt).group(1)
            question = quest_gex.search(prompt).group(1)
            new_prompt_str = f"question: {question} context: {text} </s>" 
            new_prompt = {"prompt": new_prompt_str, "correct_value": pd['correct_value']}
            new_data.append(new_prompt)
        with open(new_p, "w") as f1:
            json.dump(new_data, f1)
        self.prompt_file = str(new_p) 

    


    