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

from agent_patient_experiment import AgentPatientExperiment


from metrics import accuracy_report
from hf_tools.hf import HuggingfaceRunFxn

def main(args):
    main_dir = pathlib.Path("/home/estengel/child-lm")

    prompt_file = main_dir.joinpath("data", "agent_patient" , args.prompt_file)
    if not prompt_file.exists():
        raise AssertionError(f"Requested file: {prompt_file} does not exist")

    exp_name = args.prompt_file.split(".")[0]
    wrapper_fxn = HuggingfaceRunFxn(args.hf_model_name, device=args.device, constrained=False)
    experiment  = AgentPatientExperiment(args.model_name, exp_name, prompt_file, wrapper_fxn, 1, None)
    experiment.run(rate_limit_delay=None, overwrite=True)
    df = experiment.format_results()
    df.to_csv(main_dir.joinpath(f"{args.out_dir}/{args.model_name}_{exp_name}.csv"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", action = ActionConfigFile)

    parser.add_argument("--hf-model-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", default="results", type=str) 
    args = parser.parse_args()

    main(args)
