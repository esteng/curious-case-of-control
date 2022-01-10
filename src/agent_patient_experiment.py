from typing import Any, Dict
import time 
import pdb 
import json 
import re

import pandas as pd 
from tqdm import tqdm 

from api_tools import FixedGPTPrompt, run_experiment
from metrics import StringMetric
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
        df = pd.DataFrame(columns=['model', 'exp_type', 'prompt', 'pred_value', 'true_value', 'response'], dtype=object)
        for line in self.results:
            df = df.append(line, ignore_index=True)
        return df

    def recompute(self, prompt_data): 
        for i, res_line in enumerate(self.results):
            for line in prompt_data: 
                if line['prompt'] == res_line['prompt']:
                    true_ans = line['correct_value']
                    self.results[i]['true_value'] = true_ans

                    response = res_line['response']
                    try:
                        response = response.split("Answer: \',")[1]
                    except IndexError:
                        response = re.sub("\(","[", response)
                        response = re.sub("\)","]", response)
                        response = re.sub("'",'"', response)
                        response = json.loads(response)
                        response = response[1]
                    pred_name = re.sub("[()']", "", response).strip()
                    true_name = res_line['true']
                    lookup = self.make_lookup() 
                    metric = StringMetric(lookup)
                    metric(pred_name) 
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
            metric = StringMetric(lookup) 
            metric, responses = run_experiment(self.experiment_fxn, prompt, replicants, metric, self.kwargs) 
            for resp in responses:
                inner_metric = StringMetric(lookup) 
                inner_metric(resp[1])
                acc, count, __ = inner_metric.get_metric(true_value)
                pred = [k for k,v in count.items() if v > 0][0]
                results_line = {"model": self.model_name, "exp_type": self.exp_type,  "true_value": true_value, "pred_value": pred, "prompt": prompt, "response": resp}
                self.results.append(results_line)

                num_run_in_time += 1
                end_time = time.time()
                if rate_limit_delay is not None and num_run_in_time > rate_limit_count:
                    # need to pause to not get kicked off 
                    num_seconds_run = end_time - start_time 
                    time_to_sleep = rate_limit_delay - num_seconds_run
                    # add 10 second fudge factor just in case 
                    time.sleep(time_to_sleep + 10)
                    # reset 
                    num_run_in_time = 0 
                    start_time = time.time() 



