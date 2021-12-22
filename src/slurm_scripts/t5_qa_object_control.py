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
correct_index = 1
nicknames = json.load(open(main_dir.joinpath("data/nicknames.json")))

from hf_tools.hf import HuggingfaceRunFxn
import os
os.environ['TRANSFORMERS_CACHE'] = "/brtx/601-nvme1/estengel/.cache"


wrapper_fxn = HuggingfaceRunFxn("valhalla/t5-base-qa-qg-hl", "cuda:0")
# wrapper_fxn = None

t5_object_control_experiment  = Experiment("t5", "object-control", FixedT5Prompt, wrapper_fxn, 1, None)
#t5_object_control_experiment.recover("../results/t5_qa_object_control.csv")
t5_object_control_experiment.run(names, correct_index, verbs, actions, do_swap = True, nicknames=nicknames, rate_limit_delay=None, overwrite=True)
t5_object_control_experiment.recompute(nicknames)

t5_df = t5_object_control_experiment.format_results()

t5_df.to_csv("../results/t5_qa_object_control.csv")

#old_t5_df = pd.read_csv("../results/t5_object_control.csv")

