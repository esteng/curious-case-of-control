from typing import Dict, List
import re
import pdb 

import pandas as pd 
class Metric:
    def __init__(self, class_lookups: Dict[str, List[str]]): 
        self.class_lookups = class_lookups
        self.assert_disjoint()

    def assert_disjoint(self):
        # make sure no overlapping keywords
        for k1, keywords1 in self.class_lookups.items():
            for k2, keywords2 in self.class_lookups.items():
                if k1 == k2:
                    continue
                try:
                    assert(len(set(keywords1) & set(keywords2)) == 0)
                except AssertionError:
                    print(f"Overlapping keywords between {k1} and {k2}: {set(keywords1) & set(keywords2)}")

class StringMetric(Metric):
    def __init__(self, class_lookups: Dict[str, List[str]]):
        super().__init__(class_lookups)
        self.classes = {k: [] for k in class_lookups.keys()}
        self.classes['other'] = []

    def __call__(self, text: str):
        if type(text) is not str:
            self.classes['other'].append("Error")
        else:
            text = self.extract_answer_string(text, self.classes.keys())
            # text = text.split("\n")[0]
            # print(text)
            split_text = re.split("\s+", text.lower())
            for k, keywords in self.class_lookups.items():
                for kw in keywords:
                    if kw in split_text: 
                        self.classes[k].append(text)
                        return
            self.classes['other'].append(text)

    def get_metric(self, true_class):
        counts = {k: len(v) for k,v in self.classes.items()}
        all_counts = sum(counts.values())
        accuracy = counts[true_class]/all_counts 
        return accuracy, counts, self.classes

    def extract_answer_string(self, text, names):
        # Rule 1: if the answer is just one word, return that 
        words = re.split("\s+", text)
        if len(words) == 1:
            return words[0] 

        # Rule 2: if "Answer: NAME" appears in the text, extract that 
        name_str = [f"({n})" for n in names]
        name_str = "|".join(name_str)
        answer_text = re.findall(f"Answer:\s+({name_str})", text, flags=re.IGNORECASE) 
        answer_text = [y  for x in answer_text for y in x]
        answer_text = list(set(answer_text))
        answer_text = [x for x in answer_text if x != '']
        if len(answer_text) == 1:
            # print(text)
            # print(answer_text)
            return answer_text[0]
        elif len(answer_text) == 2:
            # tiebreaker: what do you do if both are answered 
            return "other"
        else:
            pass 
        
        # Rule 3: if there are multiple lines, return the first line 
        lines = re.split("\n+", text)
        return lines[0]
    

class LogprobMetric(Metric):
    def __init__(self, class_lookups: Dict[str, List[str]]):
        super().__init__(class_lookups)
        self.classes = {k: [] for k in class_lookups.keys()}

    def __call__(self, logprobs_sequence: List[Dict]):
        pass
        #split_text = re.split("\s+", text)
        #for k, keywords in self.class_lookups.items():
        #    for kw in keywords:
        #        if kw in split_text: 
        #            self.classes[k].append(text)
        #            return

def get_accuracy(df): 
    if len(df) == 0:
        return -1, 0, -1, 0
    total_correct = df[df['true'] == df['pred']]
    total_acc = len(total_correct)/len(df)

    # ignoring "other"
    df_no_other = df[df['pred'] != 'other']
    if len(df_no_other) == 0:
        total_acc_no_other = -1 
    else:
        total_correct_no_other = df_no_other[df_no_other['true'] == df_no_other['pred']]
        total_acc_no_other = len(total_correct_no_other)/len(df_no_other)


    return total_acc, len(df), total_acc_no_other, len(df_no_other)

def accuracy_report(df):
    total_acc = get_accuracy(df)

    all_name1s = set(df['name1'])
    all_name2s = set(df['name2'])

    acc_by_swap = {}
    for swap_val in [True, False]:
        if swap_val not in df['swap_names']:
            swap_val = str(swap_val)
        df_by_swap_val = df[df['swap_names'] == swap_val]
        acc_by_swap_val = get_accuracy(df_by_swap_val)
        acc_by_swap[swap_val] = acc_by_swap_val

    acc_by_name = {}
    for name1 in all_name1s:
        for name2 in all_name2s:
            # don't repeat, it'll be same 
            if f"{name1},{name2}" in acc_by_name.keys() or f"{name2},{name1}" in acc_by_name.keys():
                continue

            df_by_name1_name2 = df[(df['name1'] == name1) & (df['name2'] == name2)]
            df_by_name2_name1 = df[(df['name2'] == name1) & (df['name1'] == name2)]
            full_name_df = pd.concat([df_by_name1_name2, df_by_name2_name1])
            acc_name1_name2 = get_accuracy(full_name_df)
            acc_by_name[f"{name1},{name2}"] = acc_name1_name2 

    acc_by_first_name = {}
    for name1 in all_name1s:
        for name2 in all_name2s:
            df_by_name1_name2 = df[(df['name1'] == name1) & (df['name2'] == name2)]
            acc_name1_name2 = get_accuracy(df_by_name1_name2)
            acc_by_first_name[f"{name1},{name2}"] = acc_name1_name2 


    all_actions = set(df['action'])
    acc_by_action = {}
    for action in all_actions:
        df_by_action = df[df['action'] == action]
        acc_by_action[action] = get_accuracy(df_by_action)

    acc_by_verb = {}
    all_verbs = set(df['verb'])
    for verb in all_verbs:
        df_by_verb  = df[df['verb'] == verb]
        acc_by_verb[verb] = get_accuracy(df_by_verb)

    acc_by_action_by_verb = {}

    for action in all_actions:
        for verb in all_verbs:
            df_by_action_by_verb  = df[(df['action'] == action) & (df['verb'] == verb)]
            acc_by_action_by_verb[f"{action},{verb}"] = get_accuracy(df_by_action_by_verb)

    dicts = [acc_by_name, acc_by_action, acc_by_verb, acc_by_action_by_verb]
    for i, d in enumerate(dicts):
        new_d = {k:v for k,v in d.items() if v[0] > -1}
        dicts[i] = new_d

    acc_by_name, acc_by_action, acc_by_verb, acc_by_action_by_verb = dicts
    return {"total": total_acc,
            "acc_by_swap": acc_by_swap,  
            "acc_by_name": acc_by_name, 
            "acc_by_first_name": acc_by_first_name,
            "acc_by_action": acc_by_action, 
            "acc_by_verb": acc_by_verb, 
            "acc_by_action_by_verb": acc_by_action_by_verb} 
    