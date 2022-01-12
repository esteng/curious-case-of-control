import re 
import json 
import pathlib
from numpy.core.shape_base import _arrays_for_stack_dispatcher 

import seaborn as sns 
from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np 

from experiment import Experiment
from agent_patient_experiment import AgentPatientExperiment
from metrics import accuracy_report

def get_palette(): 
    return {"gpt3": '#0173b2', "gpt-neo-1.3b": '#de8f05', "gpt-neo-2.7b": '#029e73', 
            "gpt-j": '#d55e00', "jurassic-large": '#cc78bc', "jurassic-jumbo": '#ca9161', 
            "t5": '#fbafe4', "t0": '#949494'} 

def barplot(csv_groups, level, x_name, hue_name, title=None, ax=None, filtered = False, ignore_first_only = False):
    # consolidate data from different csvs 
    all_dfs = {g:[] for g in csv_groups.keys()}
    for group, csvs in csv_groups.items():
        for csv in csvs:
            try:
                df = pd.read_csv(csv)
                # print(csv, len(df))
                all_dfs[group].append(df)
            except FileNotFoundError:
                print(f"Not found: {csv}")


    reports = {g: [accuracy_report(df, ignore_first_only, total_only=True) for df in dfs] for g, dfs in all_dfs.items()}
    # data = [report[level] for report in reports]
    data = {g: [report[level] for report in group_reports] for g, group_reports in reports.items()}
    df_to_plot = pd.DataFrame(columns=["model", "acc", "type"], dtype=object)
    for group, group_data in data.items():
        for i, data_dict in enumerate(group_data):
            df = all_dfs[group][i]
            model_name = df['model'][0]
            model_name = re.sub("_", "-", model_name)
            model_name = re.sub("-qa", "", model_name)
            if type(data_dict) == dict:
                for k, v in data_dict.items():
                    if filtered: 
                        acc = float(v[2])
                    else:
                        acc = float(v[0])
                    type_name = k
                    df_to_plot = df_to_plot.append({"model": model_name, "acc": acc, "type": type_name, "group": group}, ignore_index=True)
            else:
                # v, __, __, __ = data_dict
                v= data_dict
                if filtered: 
                    acc = float(v[2])
                else:
                    acc = float(v[0])
                type_name = "total"
                df_to_plot = df_to_plot.append({"model": model_name, "acc": acc, "type": type_name, "group": group}, ignore_index=True)
    palette = get_palette()

    order = [ "gpt-neo-1.3b", "gpt-neo-2.7b", "gpt-j", "gpt3", "jurassic-large","jurassic-jumbo","t5", "t0"]

    g = sns.catplot(data = df_to_plot, kind='bar', x = x_name, y = 'acc', hue = hue_name, palette=palette, col="group", hue_order=order)
    g.set(ylim=(0, 1.0))

    # g.ax.set_ylim(top=1.0, bottom=0.0)
    # [plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
    if title is not None:
        g.fig.suptitle(title)
    for ax_line in g.axes:
        for ax in ax_line:
            xs = (-0.4, 0.4)
            ys = (0.5, 0.5)
            ax.plot(xs, ys, "-")
    return g 


def recompute(csvs, 
              names, 
              nicknames, 
              exp_name="object-control", 
              results_path = "results_to_plot", 
              use_action = False, 
              use_verb=False,
              correct_idx=None):
    prompt=None
    wrapper_fxn=None
    accuracy_reports = {}

    for csv, name in zip(csvs, names): 
        try:
            exp  = Experiment(name, exp_name, prompt, wrapper_fxn, 1, None)
            filename = pathlib.Path(csv).name
            exp.recover(csv)
            # print(csv)
            exp.recompute(nicknames, use_action=use_action, use_verb=use_verb, correct_idx=correct_idx)
            df = exp.format_results()
            report = accuracy_report(df, total_only=True)
            accuracy_reports[name] = report['total']
            try:
                df.to_csv(f"../{results_path}/{filename}")
            except FileNotFoundError:
                print(f"FileNotFound: ../{results_path}/{filename}")
        except FileNotFoundError:
            print(f"FileNotFound: {csv}")
        
    return accuracy_reports

def recompute_agent_patient(csvs, 
                            prompt_files,
                            names, 
                            results_path = "agent_patient_results_to_plot"): 
    prompt=None
    wrapper_fxn=None
    accuracy_reports = {}

    for csv, name, prompt_file in zip(csvs, names, prompt_files): 
        try:
            exp  = AgentPatientExperiment(name, "", prompt, wrapper_fxn, 1, None)
            filename = pathlib.Path(csv).name
            exp.recover(csv)
            with open(prompt_file) as f1:
                prompt_data = json.load(f1)
            assert(len(exp.results) == len(prompt_data))


            # exp.recompute(prompt_data)
            exp.recompute() 
            df = exp.format_results()
            report = accuracy_report(df, total_only=True)
            accuracy_reports[name] = report['total']
            try:
                df.to_csv(f"../{results_path}/{filename}")
            except FileNotFoundError:
                print(f"FileNotFound: ../{results_path}/{filename}")
        except FileNotFoundError:
            print(f"FileNotFound: {csv}")
        
    return accuracy_reports