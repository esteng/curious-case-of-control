from typing import Any, Dict
import time 
import pdb 

import pandas as pd 
from tqdm import tqdm 

from api_tools import FixedGPTPrompt, run_experiment
from metrics import StringMetric

class Experiment:
    def __init__(self, 
                model_name: str,    
                exp_type: str,
                prompt: FixedGPTPrompt, 
                experiment_fxn: Any, 
                replicants: int, 
                kwargs: Dict):

        self.model_name = model_name 
        self.exp_type = exp_type
        self.prompt = prompt
        self.experiment_fxn = experiment_fxn
        self.replicants = replicants 
        self.kwargs = kwargs 

        self.results = []    

    def format_results(self):
        df = pd.DataFrame(columns=['model', 'exp_type', 'name1', 'name2', 'swap_names', 'verb', 'action', 'true', 'pred', 'prompt', 'response'], dtype=object)
        for line in self.results:
            df = df.append(line, ignore_index=True)
        return df

    def run(self, names, correct_name_idx, verbs, actions, do_swap=True, qa_pair = ("Question", "Answer")):
        results_list = []
        num_run_in_time = 0
        start_time = time.time()
        swap_names_choices = [True, False]
        if not do_swap:
            # we don't need to swap for T5 since the prompt is different 
            swap_names_choices = [False]
        for n1, n2 in tqdm(names):
            correct_name = [n1, n2][correct_name_idx]
            lookup = {n1: [n1.lower()], n2: [n2.lower()]}

            for verb in verbs:
                for inf, past in actions:
                    for swap_names in swap_names_choices:
                        prompt = self.prompt(n1, n2, verb, inf, past, swap_names = swap_names, qa_words=qa_pair)
                        replicants = 1
                        text = str(prompt)
                        metric = StringMetric(lookup) 
                        metric, responses = run_experiment(self.experiment_fxn, text, replicants, metric, self.kwargs)

                        for i in range(replicants):
                            resp = responses[i] 
                            inner_metric = StringMetric(lookup) 
                            inner_metric(resp[1])
                            acc, count, __ = inner_metric.get_metric(correct_name)
                            pred = [k for k,v in count.items() if v > 0][0]

                            results_line = {"model": self.model_name, "exp_type": self.exp_type, "name1": n1, "name2": n2, "swap_names": swap_names,
                                            "verb": verb, "action": inf, "true": correct_name, "pred": pred, "prompt": text, "response": resp}
                            self.results.append(results_line)
                            num_run_in_time += 1

                            end_time = time.time()
                            if num_run_in_time > 55:
                                # need to pause to not get kicked off 
                                num_seconds_run = end_time - start_time 
                                time_to_sleep = 60 - num_seconds_run
                                # add 10 second fudge factor just in case 
                                time.sleep(time_to_sleep + 10)
                                # reset 
                                num_run_in_time = 0 
                                start_time = time.time() 

