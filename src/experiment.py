from typing import Any, Dict

import pandas as pd 

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
        df = pd.DataFrame(columns=['model', 'exp_type', 'verb', 'action', 'true', 'pred'], dtype=object)
        for line in self.results:
            df = df.append(line, ignore_index=True)
        return df

    def run(self, names, correct_name_idx, verbs, actions):
        results_list = []
        for n1, n2 in names:
            correct_name = [n1, n2][correct_name_idx]
            lookup = {n1: [n1.lower()], n2: [n2.lower()]}

            for verb in verbs:
                for inf, past in actions:
                    prompt = self.prompt(n1, n2, verb, inf, past)
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

                        results_line = {"model": self.model_name, "exp_type": self.exp_type, "verb": verb, "action": inf, "true": correct_name, "pred": pred}
                        self.results.append(results_line)