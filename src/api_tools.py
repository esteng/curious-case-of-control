
from json.decoder import JSONDecodeError
import os
import requests
import json 
from typing import List, Dict, Tuple
import re
from tqdm import tqdm
import pdb 

import numpy as np
import pathlib 
np.random.seed(12)

path_to_file = pathlib.Path(__file__).absolute() 

oai_api_key = open(path_to_file.parent.joinpath('oai_api.key')).read().strip()
hf_api_key = open(path_to_file.parent.joinpath('hf_api.key')).read().strip()
jurassic_api_key = open(path_to_file.parent.joinpath('jurassic_api.key')).read().strip() 

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
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str,
                 swap_names: bool,
                 long_instructions: bool = False,
                 prompt_hacking: bool = False,
                 just_prompt_agent: bool = False,
                 just_prompt_patient: bool = False,
                 qa_words: Tuple[str] = ("Question", "Answer"),
                 sent_or_context: str = "context",
                 passive: bool = False):

        sent_or_context_upper = [c for c in sent_or_context]
        sent_or_context_upper[0] =  sent_or_context_upper[0].upper()
        sent_or_context_upper = "".join(sent_or_context_upper)

        question_word, answer_word = qa_words

        if long_instructions:
            do_swap = np.random.choice([True, False])
            if do_swap and swap_names:
                prompt_name1, prompt_name2 = name2, name1
            else:
                prompt_name1, prompt_name2 = name1, name2
            long_instructions_str = f" Answer the question with either \"{prompt_name1}\" or \"{prompt_name2}\"."
            or_clause = f", {prompt_name1} or {prompt_name2}"

        else:
            long_instructions_str = ""
            or_clause = ""

        if not passive:
            context = f"{name1} {verb} {name2} {infinitive}."
            hack_name1, hack_name2 = name1, name2
        else:
            context = f"{name1} was {verb} by {name2} {infinitive}."
            hack_name1, hack_name2 = name2, name1

        if verb == "promised": 
            patient_question = f"Who was {verb} something{or_clause}"
        else:
            patient_question = f"Who was {verb} {infinitive}{or_clause}"
        agent_question = f"Who {verb} someone {infinitive}{or_clause}"

        if just_prompt_agent or just_prompt_patient:
            if just_prompt_patient and just_prompt_agent:
                raise AssertionError("Can't have both just_prompt_agent and just_prompt_patient")

            if just_prompt_agent: 
                context = [f"""You will be given a {sent_or_context} and a question.{long_instructions_str}\n{sent_or_context_upper}: {context}\n""",
                                f"{question_word}: {agent_question}?"]
            elif just_prompt_patient: 
                context = [f"""You will be given a {sent_or_context} and a question.{long_instructions_str}\n{sent_or_context_upper}: {context}\n""",
                                f"{question_word}: {patient_question}?"]
        else:
            if not prompt_hacking: 
                context = [f"""You will be given a {sent_or_context} and a question.{long_instructions_str}\n{sent_or_context_upper}: {context}\n""",
                            f"{question_word}: Who {past}{or_clause}?"]
            else: 
                agent_first = np.random.choice([True, False])
                if agent_first:
                    context = [f"""You will be given a {sent_or_context} and a question.{long_instructions_str}\n{sent_or_context_upper}: {context}\n""",
                                f"{question_word}: {agent_question}?",
                                f"{answer_word}: {hack_name1}",
                                f"{question_word}: {patient_question}?",
                                f"{answer_word}: {hack_name2}",
                                f"{question_word}: Who {past}{or_clause}?"]
                else:
                    context = [f"""You will be given a {sent_or_context} and a question.{long_instructions_str}\n{sent_or_context_upper}: {context}\n""",
                                f"{question_word}: {patient_question}?",
                                f"{answer_word}: {hack_name2}",
                                f"{question_word}: {agent_question}?",
                                f"{answer_word}: {hack_name1}",
                                f"{question_word}: Who {past}{or_clause}?"]
        prompt_text = f"{answer_word}: "
        self.prompt = GPTPrompt(context, prompt_text)

    def __str__(self):
        # print(str(self.prompt))
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
                 past: str,
                 swap_names: bool,
                 long_instructions: bool = False,
                 prompt_hacking: bool = False,
                 just_prompt_agent: bool = False,
                 just_prompt_patient: bool = False,
                 qa_words: Tuple[str] = ("Question", "Answer"),
                 sent_or_context: str = "context"):
        super().__init__(name1=name1,
                         name2=name2,
                         verb=verb,
                         infinitive=infinitive,
                         past=past,
                         swap_names=swap_names,
                         long_instructions=long_instructions,
                         prompt_hacking=prompt_hacking,
                         just_prompt_agent=just_prompt_agent,
                         just_prompt_patient=just_prompt_patient,
                         qa_words=qa_words,
                         sent_or_context=sent_or_context,
                         passive=False)
        
    
class FixedPassiveGPTPrompt(FixedPrompt):
    """
    Fixed prompt for passives 
    """
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str,
                 swap_names: bool,
                 long_instructions: bool = False,
                 prompt_hacking: bool = False, 
                 just_prompt_agent: bool = False,
                 just_prompt_patient: bool = False,
                 qa_words: Tuple[str] = ("Question", "Answer"),
                 sent_or_context: str = "context"):

         super().__init__(name1=name1,
                         name2=name2,
                         verb=verb,
                         infinitive=infinitive,
                         past=past,
                         swap_names=swap_names,
                         long_instructions=long_instructions,
                         prompt_hacking=prompt_hacking,
                         just_prompt_agent=just_prompt_agent,
                         just_prompt_patient=just_prompt_patient,
                         qa_words=qa_words,
                         sent_or_context=sent_or_context,
                         passive=True)

class FixedPassiveT5Prompt(FixedPrompt):
    """
    fixed prompt for T5 passives 
    """
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str,
                 swap_names: bool,
                 qa_words: Tuple[str] = None,
                 long_instructions: bool = False,
                 prompt_hacking: bool = False,
                 just_prompt_agent: bool = False,
                 just_prompt_patient: bool = False,
                 sent_or_context: str = "context"):
        super().__init__(name1=name1,
                         name2=name2,
                         verb=verb,
                         infinitive=infinitive,
                         past=past,
                         swap_names=swap_names,
                         long_instructions=long_instructions,
                         prompt_hacking=prompt_hacking,
                         just_prompt_agent=just_prompt_agent,
                         just_prompt_patient=just_prompt_patient,
                         qa_words=qa_words,
                         sent_or_context=sent_or_context,
                         passive=True)

        if verb == "promised": 
            patient_question = f"Who was {verb} something?"
        else:
            patient_question = f"Who was {verb} {infinitive}?"
        agent_question = f"Who {verb} someone {infinitive}?"

        context = [f"{name1} was {verb} by {name2} {infinitive}."]
        if not just_prompt_agent and not just_prompt_patient:
            prompt_text = f"Who {past}?"
        elif just_prompt_agent:
            prompt_text = agent_question
        elif just_prompt_patient: 
            prompt_text = patient_question
        else:
            pass
        self.prompt = T5Prompt(context, prompt_text)  
class FixedT5Prompt(FixedPrompt):
    """
    Fixed prompt for T5 questions, which are different from GPT and Jurassic
    """
    def __init__(self,
                 name1: str,
                 name2: str,
                 verb: str,
                 infinitive: str,
                 past: str,
                 swap_names: bool,
                 qa_words: Tuple[str] = None,
                 long_instructions: bool = False,
                 prompt_hacking: bool = False,
                 just_prompt_agent: bool = False,
                 just_prompt_patient: bool = False,
                 sent_or_context: str = "context"):

        super().__init__(name1=name1,
                        name2=name2,
                        verb=verb,
                        infinitive=infinitive,
                        past=past,
                        swap_names=swap_names,
                        long_instructions=long_instructions,
                        prompt_hacking=prompt_hacking,
                        just_prompt_agent=just_prompt_agent,
                        just_prompt_patient=just_prompt_patient,
                        qa_words=qa_words,
                        sent_or_context=sent_or_context,
                        passive=True)

        if verb == "promised": 
            patient_question = f"Who was {verb} something?"
        else:
            patient_question = f"Who was {verb} {infinitive}?"
        agent_question = f"Who {verb} someone {infinitive}?"

        context = [f"{name1} {verb} {name2} {infinitive}."]
        if not just_prompt_agent and not just_prompt_patient:
            prompt_text = f"Who {past}?"
        elif just_prompt_agent:
            prompt_text = agent_question
        elif just_prompt_patient: 
            prompt_text = patient_question
        else:
            pass
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

def run_ai21_jumbo_prompt(text, kwargs):
    json_prompt = {"prompt": text, "numResults": 1, "stopSequences": [".","\n"], "topKReturn": 0}
    json_prompt.update(kwargs)
    response = requests.post("https://api.ai21.com/studio/v1/j1-jumbo/complete",
        headers={f"Authorization": f"Bearer {jurassic_api_key}"},
        json=json_prompt,
    )
    data = response.json()
    #print(data)
    try:
        return data['completions'][0]['data']['text']
    except KeyError:
        pdb.set_trace() 

def run_ai21_prompt(text, kwargs):
    json_prompt = {"prompt": text, "numResults": 1, "stopSequences": [".","\n"], "topKReturn": 0}
    json_prompt.update(kwargs)
    response = requests.post("https://api.ai21.com/studio/v1/j1-large/complete",
        headers={f"Authorization": f"Bearer {jurassic_api_key}"},
        json=json_prompt,
    )
    data = response.json()
    #print(data)
    try:
        return data['completions'][0]['data']['text']
    except KeyError:
        pdb.set_trace() 
    

def run_t5_prompt(text, kwargs):
    API_URL = "https://api-inference.huggingface.co/models/valhalla/t5-base-qa-qg-hl"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        try:
            return response.json()
        except JSONDecodeError:
            # print(f"response error {response}")
            return {"response": response}

    output = query({
        "inputs": text, 
    })
    try:
        return output[0]['generated_text']
    except KeyError:
        # pdb.set_trace()
        return output 

def run_experiment(run_fxn, text, replicants, metric, kwargs, n1=None, n2=None):
    responses = []
    if replicants > 1:
        for i in tqdm(range(replicants)):
            if "Log" in str(run_fxn): 
                resp = run_fxn(text,  n1=n1, n2=n2, kwargs=kwargs)
            else:
                resp = run_fxn(text, kwargs)
            metric(resp)
            responses.append((text, resp))
    else:
        # don't do tqdm if only 1, it's annoying 
        for i in range(replicants):
            if "Log" in str(run_fxn): 
                resp = run_fxn(text,  n1=n1, n2=n2, kwargs=kwargs)
            else:
                resp = run_fxn(text, kwargs)
            metric(resp)
            responses.append((text, resp))
    return metric, responses

