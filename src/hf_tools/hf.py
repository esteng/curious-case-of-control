import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration 
import pdb 

import time 


class HuggingfaceRunFxn:
    def __init__(self, model_name, constrained = False, device="cpu"): 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, max_length=57)

        self.device = device 
        self.model.to(device)
        self.constrained = constrained

    def get_names(self, text): 
        # get names from the prompt 
        n1, n2 = re.findall("Answer the question with either \"(\w+)\" or \"(\w+)\"", text)[0]
        return n1, n2

    def __call__(self, text, kwargs):
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        input_ids = input_ids.to(self.device)
        outputs =  self.model.generate(input_ids)
        if not self.constrained:
            output_text = self.tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True)
        else:
            n1, n2 = self.get_names(text)
            pdb.set_trace() 


        return output_text 




if __name__ == "__main__":
    fxn = HuggingfaceRunFxn("valhalla/t5-base-qa-qg-hl")

    prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
Context: Avery was convinced by Joseph to go.

Question:  Who went, Avery or Joseph?
Answer: """

    t0 = time.time()
    print(fxn(prompt))
    t1 = time.time()
    print(f"on CPU took: {t1 - t0}")


    fxn = HuggingfaceRunFxn("valhalla/t5-base-qa-qg-hl", device="cuda:0")

    prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
Context: Avery was convinced by Joseph to go.

Question:  Who went, Avery or Joseph?
Answer: """

    t0 = time.time()
    print(fxn(prompt))
    t1 = time.time()
    print(f"on GPU took: {t1 - t0}")
