import logging
import json 
import pathlib 
import re 
import os
os.environ['TRANSFORMERS_CACHE'] = "/brtx/601-nvme1/estengel/.cache"
logging.getLogger("imported_module").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

from jsonargparse import ArgumentParser, ActionConfigFile

from experiment import Experiment
from api_tools import (FixedGPTPrompt, 
                       FixedPassiveGPTPrompt, 
                       FixedT5Prompt, 
                       FixedPassiveT5Prompt, 
                       run_ai21_prompt, 
                       run_gpt_prompt, 
                       run_t5_prompt)


from metrics import accuracy_report
from hf_tools.hf import HuggingfaceRunFxn

def choose_prompt(name):
    if name == "gpt-passive":
        return FixedPassiveGPTPrompt
    elif name == "gpt": 
        return FixedGPTPrompt
    elif name == "t5":
        return FixedT5Prompt
    elif name == "t5-passive": 
        return FixedPassiveT5Prompt
    else:
        raise ValueError(f"Invalid choice: {name}") 

def main(args):
    main_dir = pathlib.Path("/home/estengel/child-lm")
    names = json.load(open(main_dir.joinpath("data", args.names_file)))
    actions = json.load(open(main_dir.joinpath("data", args.action_file)))
    nicknames = json.load(open(main_dir.joinpath("data", args.nicknames_file)))

    if args.subject_control:
        str_exp_name = "subject_control"
        exp_name = "subject-control"
        verbs = ["promised"]
        correct_index = 0
        passive_name = ""
    else:
        str_exp_name = "object_control"
        if args.passive:
            exp_name = "object-control-passive"
            
            passive_name = "_passive"
            correct_index = 0
        else:
            exp_name = "object-control"
            correct_index = 1
            passive_name = ""
        verbs = ["told", "ordered", "called upon", "reminded", "urged", "asked", "persuaded", "convinced", "forced", "pushed"]

    wrapper_fxn = HuggingfaceRunFxn(args.hf_model_name, device=args.device, constrained=False)
    experiment  = Experiment(args.model_name, exp_name, choose_prompt(args.prompt_name), wrapper_fxn, 1, None)
    experiment.run(names, 
                   correct_index, 
                   verbs, 
                   actions, 
                   do_swap = True, 
                   long_instruction=args.long_instruction,
                   prompt_hacking = args.prompt_hacking, 
                   just_prompt_agent = args.just_prompt_agent,  
                   just_prompt_patient = args.just_prompt_patient, 
                   sent_or_context = args.sent_or_context,  
                   nicknames=nicknames, 
                   rate_limit_delay=None, 
                   overwrite=True)

    df = experiment.format_results()
    df.to_csv(main_dir.joinpath(f"{args.out_dir}/{args.model_name}{passive_name}_{str_exp_name}.csv"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", action = ActionConfigFile)

    parser.add_argument("--hf-model-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompt-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--subject-control", action="store_true")
    parser.add_argument("--prompt-hacking", action="store_true")
    parser.add_argument("--just-prompt-agent", action="store_true")
    parser.add_argument("--just-prompt-patient", action="store_true")
    parser.add_argument("--long-instruction", action="store_true")
    parser.add_argument("--sent-or-context", type=str, default="context")
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--names-file", type=str, default="names_top_2.json")
    parser.add_argument("--action-file", default="verbs.json")
    parser.add_argument("--nicknames-file", default="nicknames.json")
    parser.add_argument("--out-dir", default="results", type=str) 
    args = parser.parse_args()

    main(args)