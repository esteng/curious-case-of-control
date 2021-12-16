from re import I
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration 

import time 


class HuggingfaceRunFxn:
    def __init__(self, model_name, device="cpu"): 
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device 
        self.model.to(device)


    def __call__(self, text):
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        input_ids = input_ids.to(self.device)
        outputs =  self.model.generate(input_ids)
        output_text = self.tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True)
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