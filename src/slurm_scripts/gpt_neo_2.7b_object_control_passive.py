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
verbs = ["told", "ordered", "called upon", "reminded", "urged", "asked", "persuaded", "convinced", "forced", "pushed"]
actions = json.load(open(main_dir.joinpath("data/verbs.json")))
correct_index = 0
nicknames = json.load(open(main_dir.joinpath("data/nicknames.json")))

from hf_tools.hf import HuggingfaceRunFxn
import os
os.environ['TRANSFORMERS_CACHE'] = "/brtx/601-nvme1/estengel/.cache"

wrapper_fxn = HuggingfaceRunFxn("EleutherAI/gpt-neo-2.7B", device="cuda:0", constrained=False)

passive_gpt_neo_27Bobject_control_experiment  = Experiment("gpt_neo_2.7b", "object-control-passive", FixedPassiveGPTPrompt, wrapper_fxn, 1, None)

passive_gpt_neo_27Bobject_control_experiment.run(names, correct_index, verbs, actions, do_swap = True, nicknames=nicknames, rate_limit_delay=None, overwrite=True)

passive_gpt_neo_27Bdf = passive_gpt_neo_27Bobject_control_experiment.format_results()

passive_gpt_neo_27Bdf.to_csv(main_dir.joinpath("results/gpt_neo_2.7b_passive_object_control.csv"))

accuracy_report(passive_gpt_neo_27Bdf)


