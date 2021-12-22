import json
import pandas as pd 
import sys
sys.path.insert(0, "." )

from experiment import Experiment
from api_tools import (FixedGPTPrompt, 
                       FixedPassiveGPTPrompt, 
                       FixedT5Prompt, 
                       FixedPassiveT5Prompt, 
                       run_ai21_prompt, 
                       run_gpt_prompt, 
                       run_t5_prompt)

from metrics import accuracy_report
import pathlib 
main_dir = pathlib.Path("/home/estengel/child-lm")

names = json.load(open(main_dir.joinpath("data/names_top_2.json")))
verbs = ["promised"]
actions = json.load(open(main_dir.joinpath("data/verbs.json")))
correct_index = 0
nicknames = json.load(open(main_dir.joinpath("data/nicknames.json")))

from hf_tools.hf import HuggingfaceRunFxn
import os
os.environ['TRANSFORMERS_CACHE'] = "/brtx/601-nvme1/estengel/.cache"

wrapper_fxn = HuggingfaceRunFxn("EleutherAI/gpt-j-6B", device="cpu", constrained=False)

gpt_j_subject_control_experiment  = Experiment("gpt_j", "subject-control", FixedGPTPrompt, wrapper_fxn, 1, None)

gpt_j_subject_control_experiment.run(names, correct_index, verbs, actions, do_swap = True, nicknames=nicknames, rate_limit_delay=None, overwrite=True)

gpt_j_df = gpt_j_subject_control_experiment.format_results()

gpt_j_df.to_csv(main_dir.joinpath("results/gpt_j_subject_control.csv"))

accuracy_report(gpt_j_df)


