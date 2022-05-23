import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration 
from transformers import pipeline
import pdb 
import os 
import time 
import deepspeed
from parallelformers import parallelize
import torch

class HuggingfaceRunFxn:
    def __init__(self, model_name, constrained = False, device="cpu", max_len=100): 

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, max_length=max_len)
        
        self.constrained = constrained

        if device.startswith("multi"): 
            device, n_gpus = device.split("-")
            n_gpus = int(n_gpus)
        self.device = device 

        if device == "multi": 
            # run at half precision 
            self.model = self.model.half()
            device = "cuda:0"

        self.model.to(device)
        self.max_len = max_len
            
    def get_names(self, text): 
        # get names from the prompt 
        n1, n2 = re.findall("Answer the question with either \"(\w+)\" or \"(\w+)\"", text)[0]
        return n1, n2

    def __call__(self, text, kwargs):
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        if self.device != "multi": 
            input_ids = input_ids.to(self.device)
        else:
            input_ids = input_ids.to("cuda") 
        outputs =  self.model.generate(input_ids)

        if not self.constrained:
            output_text = self.tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True)
        else:
            n1, n2 = self.get_names(text)
            pdb.set_trace() 

        return output_text 



class LogProbHuggingfaceRunFxn(HuggingfaceRunFxn):
    def __init__(self, model_name, constrained = True, device="cpu", max_len=100): 
        super(LogProbHuggingfaceRunFxn, self).__init__(model_name, constrained, device, max_len)


    def __call__(self, text, n1, n2, kwargs): 
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        if self.device != "multi": 
            input_ids = input_ids.to(self.device)
        else:
            input_ids = input_ids.to("cuda") 

        n1_toks = self.tokenizer(n1, return_tensors='pt').input_ids
        n2_toks = self.tokenizer(n2, return_tensors='pt').input_ids

        n1_start_tok = n1_toks[0,0]
        n2_start_tok = n2_toks[0,0]
        try:
            outputs = self.model.forward(input_ids, output_hidden_states=False) 
            logits = outputs[0]
            n1_logit = logits[0, -1, n1_start_tok]
            n2_logit = logits[0, -1, n2_start_tok]
        except ValueError:
            outputs = self.model.generate(input_ids, output_hidden_states=False, output_scores=True, return_dict_in_generate=True) 
            logits = outputs['scores'][0]
            n1_logit = logits[0, n1_start_tok]
            n2_logit = logits[0, n2_start_tok]

        output_dict = {n1: n1_logit.detach().cpu().numpy(), 
                       n2: n2_logit.detach().cpu().numpy()}

        return output_dict

if __name__ == "__main__":
    fxn = HuggingfaceRunFxn("EleutherAI/gpt-j-6B", device='multi-2')

    prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
Context: Avery was convinced by Joseph to go.

Question:  Who went, Avery or Joseph?
Answer: """

    t0 = time.time()
    print(fxn(prompt, None))
    t1 = time.time()
    print(f"half-precision took: {t1 - t0}")

