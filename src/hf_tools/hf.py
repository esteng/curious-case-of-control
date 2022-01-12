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
            # parallelize(self.model, num_gpus=n_gpus, fp16=True, verbose='detail')
            # local_rank = int(os.getenv('LOCAL_RANK', '0'))
            # world_size = int(os.getenv('WORLD_SIZE', '1'))
            # generator = pipeline('text-generation', model=model_name,
                                #   device=local_rank)
            # ds_engine = deepspeed.init_inference(self.model,
            #                                            mp_size=2,
            #                                            dtype=torch.float,
            #                                            checkpoint=None,
            #                                            replace_method='auto')
            # self.model = ds_engine.module
            self.model = self.model.half()
            device = "cuda:0"

        # else:
        self.model.to(device)
        self.max_len = max_len
            
    def get_names(self, text): 
        # get names from the prompt 
        n1, n2 = re.findall("Answer the question with either \"(\w+)\" or \"(\w+)\"", text)[0]
        return n1, n2

    def __call__(self, text, kwargs):
        # if self.generator is None: 
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        if self.device != "multi": 
            input_ids = input_ids.to(self.device)
        else:
            input_ids = input_ids.to("cuda") 
        outputs =  self.model.generate(input_ids)
            # outputs = self.generator(text, do_sample=False, min_length = 1, max_length = self.max_len)
        # else:
        #     input_ids = self.tokenizer(text, return_tensors='pt')
        #     outputs =  self.model.generate(**input_ids)

        if not self.constrained:
            output_text = self.tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True)
        else:
            n1, n2 = self.get_names(text)
            pdb.set_trace() 


        # else:
        #     output_text = self.generator(text, do_sample=True, min_length=1, max_length=self.max_len)

        return output_text 




if __name__ == "__main__":
#     fxn = HuggingfaceRunFxn("valhalla/t5-base-qa-qg-hl")

#     prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
# Context: Avery was convinced by Joseph to go.

# Question:  Who went, Avery or Joseph?
# Answer: """

#     t0 = time.time()
#     print(fxn(prompt))
#     t1 = time.time()
#     print(f"on CPU took: {t1 - t0}")


#     fxn = HuggingfaceRunFxn("valhalla/t5-base-qa-qg-hl", device="cuda:0")

#     prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
# Context: Avery was convinced by Joseph to go.

# Question:  Who went, Avery or Joseph?
# Answer: """

#     t0 = time.time()
#     print(fxn(prompt))
#     t1 = time.time()
#     print(f"on GPU took: {t1 - t0}")

    fxn = HuggingfaceRunFxn("EleutherAI/gpt-j-6B", device='multi-2')

    prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
Context: Avery was convinced by Joseph to go.

Question:  Who went, Avery or Joseph?
Answer: """

    t0 = time.time()
    print(fxn(prompt, None))
    t1 = time.time()
    print(f"on 2 GPU took: {t1 - t0}")


#     fxn = HuggingfaceRunFxn("EleutherAI/gpt-neo-2.7B", device='multi-3')

#     prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
# Context: Avery was convinced by Joseph to go.

# Question:  Who went, Avery or Joseph?
# Answer: """

#     t0 = time.time()
#     print(fxn(prompt, None))
#     t1 = time.time()
#     print(f"on 3 GPU took: {t1 - t0}")

#     fxn = HuggingfaceRunFxn("EleutherAI/gpt-neo-2.7B", device='cpu')

#     prompt = """You will be given a context and a question. Answer the question with either "Avery" or "Joseph".
# Context: Avery was convinced by Joseph to go.

# Question:  Who went, Avery or Joseph?
# Answer: """

#     t0 = time.time()
#     print(fxn(prompt, None))
#     t1 = time.time()
#     print(f"on cputook: {t1 - t0}")
