
import os
import requests
import json 
from typing import List, Dict
import re
from tqdm import tqdm

oai_api_key = open('oai_api.key').read().strip()
hf_api_key = open('hf_api.key').read().strip()
jurassic_api_key = open('jurassic_api.key').read().strip() 

class Prompt:
    def __init__(self, 
                 context: List[str], 
                 prompt: str, 
                 context_sep = "\n"): 
        self.context = context 
        self.prompt = prompt 
        self.context_sep = context_sep

class T5Prompt(Prompt):
    def __init__(self, 
                 context: List[str], 
                 prompt: str, 
                 context_sep = "\n"): 
        super().__init__(context,
                         prompt,
                         context_sep)

    def __str__(self): 
        context_str=self.context_sep.join(self.context) 
        return f"question: {self.prompt} context: {context_str} </s>" 


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

class FixedPrompt:
    def __init__(self):
        self.prompt = None

    def __str__(self):
        return str(self.prompt)
class FixedGPTPrompt(FixedPrompt):
    """
    Fixed prompt for object and subject control 
    """
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str):
        super().__init__()
        context = [f"""You will be given a context and a question. Answer the question with either "{name1}" or "{name2}".\nContext: {name1} {verb} {name2} {infinitive}.\n""",
                    f"Question:  Who {past}, {name1} or {name2}?"]
        prompt_text = "Answer: "
        self.prompt = GPTPrompt(context, prompt_text)
    
class FixedPassiveGPTPrompt:
    """
    Fixed prompt for passives 
    """
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str):
        super().__init__()
        context = [f"""You will be given a context and a question. Answer the question with either "{name1}" or "{name2}".\nContext: {name1} was {verb} by {name2} {infinitive}.\n""",
                    f"Question:  Who {past}, {name1} or {name2}?"]
        prompt_text = "Answer: "
        self.prompt = GPTPrompt(context, prompt_text)
    
class FixedT5Prompt:
    """
    Fixed prompt for T5 questions, which are different from GPT and Jurassic
    """
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str):

                    # print(text)
        context = [f"{name1} {verb} {name2} {infinitive}."]
        prompt_text = f"Who {past}?"
        self.prompt = T5Prompt(context, prompt_text)

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


def run_ai21_prompt(text, kwargs):
    json_prompt = {"prompt": text, "numResults": 1, "stopSequences": [".","\n"], "topKReturn": 0}
    json_prompt.update(kwargs)
    response = requests.post("https://api.ai21.com/studio/v1/j1-large/complete",
        headers={f"Authorization": f"Bearer {jurassic_api_key}"},
        json=json_prompt,
    )
    data = response.json()
    #print(data)
    return data['completions'][0]['data']['text']

def run_t5_prompt(text, kwargs):
    API_URL = "https://api-inference.huggingface.co/models/valhalla/t5-base-qa-qg-hl"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text, 
    })
    try:
        return output[0]['generated_text']
    except KeyError:
        return output 

def run_experiment(run_fxn, text, replicants, metric, kwargs):
    responses = []
    for i in tqdm(range(replicants)):
        resp = run_fxn(text, kwargs)
        metric(resp)
        responses.append((text, resp))
    return metric, responses

