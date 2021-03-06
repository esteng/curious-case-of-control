from typing import Dict, List
import re
import pdb 
import Levenshtein as pylev

import pandas as pd 
import numpy as np 

ACTION_LOOKUP = {"to come": "came", "to go": "went", "to read": "read", "to run": "ran", "to call": "called"}

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

    def __call__(self, text: str, action: str = None, verb: str = None, prompt: str = None):
        if prompt is not None:
            text = self.remove_prompt(text, prompt)
        if type(text) is not str:
            self.classes['other'].append("Error")
        else:
            text = self.extract_answer_string(text, self.classes.keys(), action=action, verb=verb)
            # text = text.split("\n")[0]
            # print(text)
            split_text = re.split("\s+", text.lower())
            for k, keywords in self.class_lookups.items():
                for kw in keywords:
                    if kw in split_text: 
                        self.classes[k].append(text)
                        return
            self.classes['other'].append(text)

    def remove_prompt(self, text, prompt):
        # remove the prompt from the text, in cases where the model repeats the prompt before answering 
        # print(f"in removing prompt")
        # text = re.sub("\n", r"\n", text)
        # print(str(text))
        # print()
        # print(prompt)
        first_p_chars = text[0:len(prompt)]

        # print(f"first p chars {first_p_chars}") 
        
        if len(first_p_chars) == 0:
            return text

        distance = pylev.distance(first_p_chars, prompt)
        # print(distance)
        # print(distance / len(first_p_chars) )
        # if less than 10% difference 
        # sys.exit()
        if distance / len(first_p_chars) < 0.1:
            prompt_lines = [x for x in re.split(r"(\n)", prompt) if x is not None]
            text_lines = [x for x in re.split(r"(\n)|(\\n)", text) if x is not None]

            if len(text_lines) > len(prompt_lines):
                # print(f"removing ") 
                # print("PROMPT=========")
                # print(prompt)
                # print("TEXT===========")
                # print(text)
                text_lines = text_lines[len(prompt_lines)-1:]
                text = "\n".join(text_lines)
                # print("AFTER============")
                # print(text)
                # sys.exit()
        return text

    def get_metric(self, true_class):
        counts = {k: len(v) for k,v in self.classes.items()}
        all_counts = sum(counts.values())
        accuracy = counts[true_class]/all_counts 
        return accuracy, counts, self.classes

    def extract_answer_string(self, text, names, action=None, verb=None):
        # Rule 0: if prompt text appears in output, remove prompt text by extracting just one question
        # comes up in prompt hacking experiments 
        assert(verb is None or action is None)
        if verb is not None:
            question_gex = fr"Question:\s+Who\s+{verb},\s+\w+\s+or\s+\w+\?[\s\\n]*Answer:\s+(\w+)"
            answer  = re.search(question_gex, text)
            if answer is not None: 
                return answer.group(1)
        if action is not None:
            action = ACTION_LOOKUP[action]
            question_gex = fr"Question:\s+Who\s+{action},\s+\w+\s+or\s+\w+\?[\s\\n]*Answer:\s+(\w+)"
            answer  = re.search(question_gex, text)
            if answer is not None: 
                return answer.group(1)


        # Rule 1: if the answer is just one word, return that 
        words = re.split("\s+", text)
        if len(words) == 1:
            word = words[0]
            # remove punct
            word = re.sub("[()\.]", "", word)
            return word

        # Rule 2: if "Answer: NAME" appears in the text, extract the first occurrence of that 
        name_str = [f"({n})" for n in names]
        name_str = "|".join(name_str)
        answer_text = re.findall(f"Answer:\s+({name_str})", text, flags=re.IGNORECASE) 
        answer_text = [y  for x in answer_text for y in x]
        answer_text_set = list(set(answer_text))
        answer_text_set = [x for x in answer_text_set if x != '']
        if len(answer_text_set) == 1:
            # print(text)
            # print(answer_text)
            return answer_text_set[0]
        elif len(answer_text_set) > 1:
            # tiebreaker: what do you do if both are answered
            # return the first one  
            return answer_text[0]
        else:
            pass 
        
        # Rule 3: if there are multiple lines, return the first line; only apply if not doing prompt hacking 
        if verb is None and action is None:
            lines = re.split("\n+", text)
            return lines[0]
        return "other"
    

class LogprobMetric(Metric):
    def __init__(self, class_lookups: Dict[str, List[str]]):
        super().__init__(class_lookups)
        self.classes = {k: [] for k in class_lookups.keys()}

    def __call__(self, log_prob_dict: Dict[str, float]):
        n1, n2 = log_prob_dict.keys() 
        v1, v2 = log_prob_dict[n1], log_prob_dict[n2]
        if np.exp(v1) > np.exp(v2): 
            self.classes[n1].append(1)
        else:
            self.classes[n2].append(1)  

    def get_metric(self, true_class):
        counts = {k: len(v) for k,v in self.classes.items()}
        all_counts = sum(counts.values())
        accuracy = counts[true_class]/all_counts 
        return accuracy, counts, self.classes
        

def process_to_ignore(df, thresh_perc = 0.9):
    # pre-process DF to remove whole names where the model just picks the first name in the instructions 
    all_name1s = set(df['name1'])
    all_name2s = set(df['name2']) 
    df['swap_names'] = df['swap_names'].astype("bool")
    remove_names = []
    for n1 in all_name1s:
        for n2 in all_name2s:
            if n1 == n2:
                continue
            sub_df = df[(df['name1'] == n1) & (df['name2'] == n2)]
           
            # print(f"subdf {len(sub_df)}")
            swap_val_true, swap_val_false = True, False

            sub_df_swap = sub_df[sub_df['swap_names'] == swap_val_true]
            sub_df_no_swap = sub_df[sub_df['swap_names'] != swap_val_true]

            pred_with_swap = sub_df_swap['pred']
            pred_no_swap = sub_df_no_swap['pred']

            thresh = int(thresh_perc * len(sub_df_swap))
            # if more than 90% (or whatever thresh_perc is set to) of the examples are just 
            # the name in a certain position (first or second) then ignore than name pair
            if (pred_with_swap.eq(n2).sum() > thresh and pred_no_swap.eq(n1).sum() > thresh) or \
               (pred_with_swap.eq(n1).sum() > thresh and pred_no_swap.eq(n2).sum() > thresh) :
                # print(f"removing {n1},{n2}")
                remove_names.append((n1,n2))
                remove_names.append((n2,n1))

    # print(f"before: {len(df)}")
    for n1, n2 in remove_names:
        df = df[(df['name1'] != n1) | (df['name2'] != n2)]
    # print(f"after: {len(df)}")
    return df

class AgentPatientStringMetric(StringMetric):
    def __init__(self, class_lookups: Dict[str, List[str]]):
        super().__init__(class_lookups)

    def __call__(self, text: str, prompt: str = None): 
        if type(text) is not str:
            self.classes['other'].append("Error")
        else:
            text = self.extract_answer_string(text, prompt)
            split_text = re.split("\s+", text.lower())
            for k, keywords in self.class_lookups.items():
                for kw in keywords:
                    if kw in split_text: 
                        self.classes[k].append(text)
                        return
            self.classes['other'].append(text)


    def extract_answer_string(self, text, prompt=None):
        if prompt is not None:
            text = self.remove_prompt(text, prompt)

        # Rule 1: if the answer is just one word, return that 
        words = re.split("\s+", text)
        if len(words) == 1:
            # print(words[0])
            word = re.sub(r"[\(\)]", "", words[0])
            # word = words[0]
            # print(f"returning word {word}")
            return word

        # Rule 2: if the string "[Tt]he answer is X" appears, extract that
        ans_gex = re.compile(r"[tT]he answer is ((yes)|(no))", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            # even if there is more than one, take first 
            return answer_text.group(1)

        # Rule 3: if "The state of the participant is changed" appears return yes
        ans_gex = re.compile(r"the state of the participant[\w \"]* is changed", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            # print(f"returning because the state is : {answer_text.group(0)}")
            return "Yes"

        # Rule 4: if "The state of the participant is not changed" appears return yes
        ans_gex = re.compile(r"the state of the participant[\w \"]is not changed", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            # print(f"returning because the state is : {answer_text.group(0)}")
            return "No"

        # Rule 3: if "changes in state" appears return yes
        ans_gex = re.compile(r"changes in state", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            return "Yes"

        # Rule 3: if "changes in state" appears return yes
        ans_gex = re.compile(r"the state of [\w\s\"]+does change( in state)?\.", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            return "Yes"

        # Rule 3: if "doesn't change in state" appears return no
        ans_gex = re.compile(r"((doesn't)|(does n[o\']t)) change( in state)?", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            return "No"

        # Rule 5: look for ^((Yes)|(No))
        ans_gex = re.compile(r"^((Yes)|(No))", flags=re.IGNORECASE)
        answer_text = ans_gex.search(text)
        if answer_text is not None:
            # print(f"because it starts with {answer_text.group(1)}")
            return answer_text.group(1)

        # Rule 6: look for ((Yes)|(No))
        # ans_gex = re.compile(r"((Yes)|(No))", flags=re.IGNORECASE)
        # # remove "yes-no" so that repeated instructions don't count
        # text = re.sub("yes-no", "", text)
        # answer_text = ans_gex.search(text)
        # if answer_text is not None:
        #     counts = {"yes":0, "no": 0}
        #     for gidx in range(len(answer_text.group())):
        #         t = answer_text.group(gidx).lower()
        #         counts[t] += 1
        #     print(f"returning because ys or no in text \n{text}")
        #     return answer_text.group(1)
            # if counts['yes'] == counts['no']:
            #     return "other"
            # if counts['yes'] > counts['no']:
            #     return "Yes"
            # if counts['yes'] < counts['no']:
            #     return "No"

            # return answer_text.group(1)

        return "other"

def get_accuracy(df, ignore_first_only = False): 
    if len(df) == 0:
        return -1, 0, -1, 0
    if ignore_first_only:
        # print(f"df before: {len(df)}")
        df = process_to_ignore(df)
        if len(df) == 0:
            return -1, 0, -1, 0
        # print(f"df after: {len(df)}")


    total_correct = df[df['true'] == df['pred']]
    total_acc = len(total_correct)/len(df)

    # ignoring "other"
    df_no_other = df[df['pred'] != 'other']
    if len(df_no_other) == 0:
        total_acc_no_other = -1 
    else:
        total_correct_no_other = df_no_other[df_no_other['true'] == df_no_other['pred']]
        total_acc_no_other = len(total_correct_no_other)/len(df_no_other)

    # print(total_acc, total_acc_no_other)
    return total_acc, len(df), total_acc_no_other, len(df_no_other)

def accuracy_report(df, ignore_first_only=False, total_only = False):
    total_acc = get_accuracy(df, ignore_first_only=ignore_first_only)

    if total_only:
        return {"total": total_acc}
    all_name1s = set(df['name1'])
    all_name2s = set(df['name2'])

    acc_by_swap = {}
    for swap_val in [True, False]:
        if swap_val not in df['swap_names']:
            swap_val = str(swap_val)
        df_by_swap_val = df[df['swap_names'] == swap_val]
        acc_by_swap_val = get_accuracy(df_by_swap_val, ignore_first_only=ignore_first_only)
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
            acc_name1_name2 = get_accuracy(full_name_df, ignore_first_only=ignore_first_only)
            acc_by_name[f"{name1},{name2}"] = acc_name1_name2 

    acc_by_first_name = {}
    for name1 in all_name1s:
        for name2 in all_name2s:
            df_by_name1_name2 = df[(df['name1'] == name1) & (df['name2'] == name2)]
            acc_name1_name2 = get_accuracy(df_by_name1_name2, ignore_first_only=ignore_first_only)
            acc_by_first_name[f"{name1},{name2}"] = acc_name1_name2 


    all_actions = set(df['action'])
    acc_by_action = {}
    for action in all_actions:
        df_by_action = df[df['action'] == action]
        acc_by_action[action] = get_accuracy(df_by_action, ignore_first_only=ignore_first_only)

    acc_by_verb = {}
    all_verbs = set(df['verb'])
    for verb in all_verbs:
        df_by_verb  = df[df['verb'] == verb]
        acc_by_verb[verb] = get_accuracy(df_by_verb, ignore_first_only=ignore_first_only)

    acc_by_action_by_verb = {}

    for action in all_actions:
        for verb in all_verbs:
            df_by_action_by_verb  = df[(df['action'] == action) & (df['verb'] == verb)]
            acc_by_action_by_verb[f"{action},{verb}"] = get_accuracy(df_by_action_by_verb, ignore_first_only=ignore_first_only)

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
    