
import os
import openai
import requests
import json 
from typing import List, Dict
import re
from tqdm import tqdm

oai_api_key = open('oai_api.key').read().strip()
hf_api_key = open('hf_api.key').read().strip()


class Prompt:
    def __init__(self, 
                 context: List[str], 
                 prompt: str, 
                 context_sep = "\n"): 
        self.context = context 
        self.prompt = prompt 
        self.context_sep = context_sep

class T5Prompt(Prompt):
    pass  

class GPTPrompt(Prompt):
    def __init__(self, 
                 context: List[str], 
                 prompt: str, 
                 context_sep = "\n"): 
        super().__init__(context,
                         prompt,
                         context_sep)

    def __str__(self): 
        context_str=self.context_sep.join(self.context) 
        return f"{context_str}{self.context_sep}{self.prompt}"


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
        split_text = re.split("\s+", text.lower())
        for k, keywords in self.class_lookups.items():
            for kw in keywords:
                if kw in split_text: 
                    self.classes[k].append(text)
                    return
        self.classes['other'].append(text)

    def get_metric(self):
        counts = {k: len(v) for k,v in self.classes.items()}
        return counts, self.classes

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



def run_gpt_prompt(text, kwargs): 
    prompt = {
      "prompt": text,
    }
    if kwargs is not None:
      prompt.update(kwargs) 
    #print(prompt)

    r = requests.get("https://api.openai.com/v1/engines/davinci/completions/browser_stream",
      headers={
        "Authorization": f"Bearer {oai_api_key}"
      },
      stream=True,
      params=prompt)
    responses = []

    def read_response(line): 
        line = line.decode("utf-8") 
        try: 
            line = json.loads(line.split("data: ")[1]) 
        except:
            return None
        return line 

    for line in r:
        resp = read_response(line) 
        if resp is not None:
            responses.append(resp) 

    text = " ".join([x['choices'][0]['text'] for x in responses])
    return text 

def run_t5_prompt(text, kwargs):
    API_URL = "https://api-inference.huggingface.co/models/valhalla/t5-base-qa-qg-hl"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": "question: Is a doll wearing a blindfold easy to see, or hard to see? context: Answer the following question with \"easy\" or \"hard\"."
    })
    return output['generated_text']

def run_experiment(run_fxn, text, replicants, metric, kwargs):
    responses = []
    for i in tqdm(range(replicants)):
        resp = run_fxn(text, kwargs)
        metric(resp)
        responses.append((text, resp))
    return metric, responses

