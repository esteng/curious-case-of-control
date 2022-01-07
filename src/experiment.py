from typing import Any, Dict
import time 
import pdb 
import json 
import re

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

    def recover(self, csv):
        """
        recover a partially completed experiment 
        """
        assert(len(self.results) == 0)
        df = pd.read_csv(csv)
        self.results = df.to_dict(orient="records")

    def check_results(self, prompt):
        """
        check results to see if prompt has been done before 
        """
        for i, line in enumerate(self.results):
            if line['prompt'].strip() == prompt.strip():
                return True, i
        return False, -1

    def recompute(self, nicknames, use_action: bool = False, use_verb: bool = False, correct_idx = None):
        for i, res_line in enumerate(self.results):
            if correct_idx is not None:
                # overwrite correct
                if correct_idx == 0:
                    true_ans = res_line["name1"]
                elif correct_idx == 1: 
                    true_ans = res_line["name2"]
                else:
                    raise AssertionError(f"you can't use correct_idx={correct_idx}")
                self.results[i]['true'] = true_ans

            action, verb = None, None
            if use_action: 
                action = res_line['action']
            if use_verb:
                verb = res_line['verb']
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
            n1, n2 = res_line['name1'], res_line['name2']
            lookup = self.make_lookup(n1, n2, nicknames)
            metric = StringMetric(lookup)
            metric(pred_name, action=action, verb=verb)
            acc, count, __ = metric.get_metric(true_name)
            pred = [k for k,v in count.items() if v > 0][0]
            self.results[i]['pred'] = pred


    def make_lookup(self, n1, n2, nicknames=None):
        if nicknames is None: 
            lookup = {n1: [n1.lower()], n2: [n2.lower()]}
        else:
            lookup = {n1: [x.lower() for x in nicknames[n1]],
                      n2: [x.lower() for x in nicknames[n2]]}
        return lookup 

    def run(self, 
            names, 
            correct_name_idx, 
            verbs, 
            actions, 
            do_swap=True, 
            qa_pair = ("Question", "Answer"), 
            long_instruction = long_instruction,
            prompt_hacking = False, 
            just_prompt_agent = False, 
            just_prompt_patient = False, 
            sent_or_context = "context", 
            overwrite=False, 
            nicknames = None, 
            rate_limit_delay = 60, 
            rate_limit_count=55):
        results_list = []
        num_run_in_time = 0
        start_time = time.time()
        swap_names_choices = [True, False]
        if not do_swap:
            # we don't need to swap for T5 since the prompt is different 
            swap_names_choices = [False]
        for n1, n2 in tqdm(names):
            correct_name = [n1, n2][correct_name_idx]
            lookup = self.make_lookup(n1, n2, nicknames)

            for verb in verbs:
                for inf, past in actions:
                    for swap_names in swap_names_choices:
                        prompt = self.prompt(n1, n2, 
                                             verb, 
                                             inf, 
                                             past, 
                                             swap_names = swap_names, 
                                             qa_words=qa_pair, 
                                             sent_or_context=sent_or_context,
                                             long_instruction=long_instruction,
                                             prompt_hacking=prompt_hacking,
                                             just_prompt_agent=just_prompt_agent,
                                             just_prompt_patient=just_prompt_patient)
                        text = str(prompt)
                        already_done, done_idx = self.check_results(text)

                        replicants = self.replicants
                        metric = StringMetric(lookup) 
                        # load things we've already looked at unless told to overwrite 
                        if already_done and not overwrite:
                            continue 
                            # response = self.results[done_idx]['response']
                            # if response[0] == "(":
                            #     response = response.split("Answer: \',")[1]
                            #     responses = [(None, re.sub("[()']", "", response))]
                            # else:
                            #     responses = [(None, response)]
                            # # each replicant is already on a different line
                            # replicants = 1
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
                            if rate_limit_delay is not None and num_run_in_time > rate_limit_count:
                                # need to pause to not get kicked off 
                                num_seconds_run = end_time - start_time 
                                time_to_sleep = rate_limit_delay - num_seconds_run
                                # add 10 second fudge factor just in case 
                                time.sleep(time_to_sleep + 10)
                                # reset 
                                num_run_in_time = 0 
                                start_time = time.time() 

