from typing import Dict, List
import re
import pdb 

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
        return -1
    total_correct = df[df['true'] == df['pred']]
    total_acc = len(total_correct)/len(df)
    return total_acc, len(df) 

def accuracy_report(df):
    total_acc = get_accuracy(df)

    all_name1s = set(df['name1'])
    all_name2s = set(df['name2'])

    acc_by_name = {}
    for name1 in all_name1s:
        for name2 in all_name2s:
            df_by_name1_name2 = df[(df['name1'] == name1) & (df['name2'] == name2)]
            acc_name1_name2 = get_accuracy(df_by_name1_name2)
            acc_by_name[f"{name1},{name2}"] = acc_name1_name2 

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
        new_d = {k:v for k,v in d.items() if v > -1}
        dicts[i] = new_d

    acc_by_name, acc_by_action, acc_by_verb, acc_by_action_by_verb = dicts
    return total_acc, acc_by_name, acc_by_action, acc_by_verb, acc_by_action_by_verb
    