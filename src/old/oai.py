import os
import openai
import requests
import json 
from typing import List

api_key = os.getenv("OPENAI_API_KEY")

def run_prompt(text, kwargs): 
    prompt = {
      "prompt": text,
    }
    #prompt = {"prompt": text,
    #        "max_tokens": 5,
    #     "temperature": 1,
    #     "top_p": 1,
    #     "n": 1,
    #     "logprobs": 5, 
    #     "stop": "\n"}
    #prompt.update(kwargs) 
    #print(prompt)

    r = requests.get("https://api.openai.com/v1/engines/davinci/completions/browser_stream",
      headers={
        "Authorization": f"Bearer {api_key}"
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


         

    #print("".join(responses)) 
    text = " ".join([x['choices'][0]['text'] for x in responses])
    return text 

class Prompt:
    def __init__(self, 
                 context: List[str], 
                 prompt: str, 
                 context_sep = "\n"): 
        self.context = context 
        self.prompt = prompt 
        self.context_sep = context_sep 

    def __str__(self): 
        context_str=self.context_sep.join(self.context) 
        return f"{context_str}{self.context_sep}{self.prompt}"

text1a =  """Q: Is a person with a blindfold on easy to see, or hard to see?\nA: """

text2a =  """Q: Is a doll with a blindfold on easy to see, or hard to see?\nA: Easy to see\nQ: Is a person with a blindfold on easy to see, or hard to see?\nA: """

text3a = """Q: Is a dog biting a cat eager to bite, or slow to bite?\nEager to bite\nQ: Is a doll with a blindfold on easy to see, or hard to see?\nA: Easy to see\nQ: Is a person with a blindfold on easy to see, or hard to see?\nA: """


text1b =  """Q: Is a person wearing a blindfold on easy to see, or hard to see?\nA: """


text2b =  """Q: Is a doll wearing a blindfold on easy to see, or hard to see?\nA: Easy to see\nQ: Is a person wearting a blindfold on easy to see, or hard to see?\nA: """



#for prompt in [text1a, text2a, text3a, text1b, text2b]: 
#    resp = run_prompt(prompt) 
#
#    print(f"PROMPT:\n{prompt}")
#    print(f"RESPONSE:\n-----------{resp}")

#prompt1c = Prompt(context=["Two people named Paul and George are playing. Paul promises George to stand on a hill."], prompt = "Q: Who stood on the hill?") 
#prompt2c = Prompt(context=["Two people named Paul and George are playing. Paul forces George to stand on a hill."], prompt = "Q: Who stood on the hill?") 
#text1c = str(prompt1c) 
#text2c = str(prompt2c) 
#
#for prompt in [text1c, text2c]: 
#    resp = run_prompt(prompt) 
#
#    print(f"PROMPT:\n{prompt}")
#    print(f"RESPONSE:\n-----------{resp}")

#prompt1d = Prompt(context=["Mary dreamt that Paul had left."], prompt = "Q: Did Paul leave?") 
#prompt2d = Prompt(context=["Two people named Paul and George are playing. Paul forces George to stand on a hill."], prompt = "Q: Who stood on the hill?") 
#text1d = str(prompt1d) 
#kwargs = {"max_tokens": 5,
#         "temperature": 1,
#         "top_p": 1,
#         "n": 1,
#         "logprobs": 5, 
#         "stop": "\n"}
#
#for prompt in [text1d]: 
#    resp = run_prompt(prompt, kwargs) 
#
#    print(f"PROMPT:\n{prompt}")
#    print(f"RESPONSE:\n-----------{resp}")
