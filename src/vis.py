import re 
import json 
import pathlib
from numpy.core.shape_base import _arrays_for_stack_dispatcher 

import seaborn as sns 
from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np 
from statsmodels.stats.contingency_tables import mcnemar

from experiment import Experiment
from agent_patient_experiment import AgentPatientExperiment
from metrics import accuracy_report

def get_palette(): 
    return {"gpt3": '#0173b2', "gpt-neo-1.3b": '#de8f05', "gpt-neo-2.7b": '#029e73', 
            "gpt-j": '#d55e00', "jurassic-large": '#cc78bc', "jurassic-jumbo": '#ca9161', 
            "t5": '#fbafe4', "t0": '#949494'} 

def barplot(csv_groups, level, x_name, hue_name, title=None, ax=None, filtered = False, ignore_first_only = False, no_t5=False):
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

    if no_t5:
        order = [ "gpt-neo-1.3b", "gpt-neo-2.7b", "gpt-j", "gpt3", "jurassic-large","jurassic-jumbo", "t0"]
    else:
        order = [ "gpt-neo-1.3b", "gpt-neo-2.7b", "gpt-j", "gpt3", "jurassic-large","jurassic-jumbo","t5", "t0"]

    g = sns.catplot(data = df_to_plot, 
                    kind='bar', 
                    x = x_name,
                     y = 'acc', 
                     hue = hue_name, 
                     palette=palette, 
                     col="group", 
                     hue_order=order,
                     legend=False)
    g.set(ylim=(0, 1.0))

    if title is not None:
        g.fig.subplots_adjust(top=0.9) 
        g.fig.suptitle(title) 
    for ax_line in g.axes:
        for ax in ax_line:
            xs = (-0.4, 0.4)
            ys = (0.5, 0.5)
            ax.plot(xs, ys, "-", color="#000000")
    g.set_titles("{col_name}", fontsize=14) 
    g.set_xticklabels([])
    g.set_xticklabels([])
    g.set(xlabel=None)
    g.set(xticks=[])
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
            try:
                assert(len(exp.results) == len(prompt_data))
            except AssertionError:
                print(csv)
                raise AssertionError()


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


def execute_mcnemar_agent_patient(model1, model1_path, model2, model2_path, is_random, condition):
    exp1  = AgentPatientExperiment(model1, "", None, None, 1, None)
    exp1.recover(model1_path)
    if not is_random:
        exp2 = AgentPatientExperiment(model2, "", None, None, 1, None) 
        exp2.recover(model2_path)
    else:
        exp2 = AgentPatientExperiment("random", "", None, None, 1, None) 
        yes_no = ["Yes", "No"]
        if model2.split("-")[1] == "yes":
            pred_val = "Yes"
        elif model2.split("-")[1] == "no":
            pred_val = "No"
        else:
            pred_val = np.random.choice(["Yes", "No"])

        if condition == "volition": 
            true_vals = "No, Yes, Yes, No, Yes, No, Yes, Yes, Yes, No, No, Yes, Yes, No, No, Yes, No, Yes, Yes, Yes, No, Yes, No, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes, No, No, Yes, Yes, Yes, Yes, Yes, Yes, No, Yes, No, Yes, No, Yes, Yes, No, No, No, No, Yes, No, Yes, No, No, No, Yes, Yes, No, Yes, No, No, Yes, No, No, Yes, Yes, Yes, No, Yes, No, Yes, No, Yes, No, No, Yes, Yes, No, No, Yes, No, Yes, Yes, No, No, Yes, No, Yes, No, Yes, No, No, Yes, Yes, No, Yes, No, Yes, Yes, Yes, Yes, Yes, No, Yes, Yes, No, Yes, No, Yes, No, Yes, Yes, No, Yes, Yes, No, Yes, No, Yes, Yes, Yes, No, Yes, Yes, Yes, No, No, Yes, Yes, Yes, No, Yes, Yes, No, No, No, Yes, No, Yes, No, Yes, No, No, Yes, Yes, No, No, No, No, Yes, Yes, No, Yes, Yes, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, No, Yes, No, No".split(", ")
            top = 168
        else:
            true_vals = "No, No, No, No, Yes, Yes, Yes, Yes, Yes, No, Yes, No, Yes, Yes, No, Yes, Yes, Yes, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, No, No, No, Yes, Yes, No, No, No, No, Yes, Yes, Yes, No, No, No, No, No, No, Yes, Yes, Yes, No, Yes, Yes, Yes, Yes, No, No, Yes, No, No, No, No, Yes, No, No, Yes, No, Yes, Yes, Yes, Yes, Yes, No, No, Yes, No, Yes, No, No, Yes, Yes, Yes, No, Yes, No, No, Yes, Yes, Yes, Yes, No, Yes, No, No, Yes, No, Yes, No, Yes, Yes, No, No, Yes, No, Yes, No, No, Yes, Yes, Yes, Yes, Yes, No, Yes, No, No, No, No, No".split(", ")
            top = 118
        exp2.results = [{"pred": pred_val, "true": true_vals[i]} for i in range(top)]


    table = {model1: {"correct": [], "incorrect": []},
            model2: {"correct": [], "incorrect": []}}
        
    for i, (m1, m2) in enumerate(zip(exp1.results, exp2.results)): 
        # drop other
        if m1['pred'] == m1['true'] and m1['pred'] != 'other':
            table[model1]['correct'].append(i)
        elif m1['pred'] != 'other': 
            table[model1]['incorrect'].append(i)
        else:
            pass 
        if m2['pred'] == m2['true'] and m2['pred'] != 'other':
            table[model2]['correct'].append(i)
        elif m2['pred'] != 'other':
            table[model2]['incorrect'].append(i)
        else:
            pass

    # ----------- table -------------------
    #               model 1
    #           |-------------|-----------|------------|
    #           |             |  correct  |  incorrect |
    #           |-------------|-----------|------------|
    # model 2   |  correct    |           |            |
    #           |incorrect    |           |            |
    #           |-------------|-----------|------------|

    table_arr = np.zeros((2,2))
    corr1_corr2 = set(table[model1]['correct']) & set(table[model2]['correct'])
    incorr1_corr2 = set(table[model1]['incorrect']) & set(table[model2]['correct'])
    corr1_incorr2 = set(table[model1]['correct']) & set(table[model2]['incorrect'])
    incorr1_incorr2 = set(table[model1]['incorrect']) & set(table[model2]['incorrect'])

    table_arr[0,0] = len(corr1_corr2)
    table_arr[1,0] = len(incorr1_corr2)
    table_arr[0,1] = len(corr1_incorr2)
    table_arr[1,1] = len(incorr1_incorr2)

    mac = mcnemar(table_arr, exact=True, correction=False)
    return mac.pvalue, mac.statistic, table

def do_mcnemar_agent_patient(model1, model2, prefix1, prefix2, two_affix1, two_affix2, condition, results_dir="agent_patient_results_to_plot"): 
    path_to_file = pathlib.Path("").absolute()
    parent = path_to_file.parent
    ## Test to see if model1 is actually better than model2 
    parent_path1 = parent.joinpath(f"{results_dir}")
    parent_path2 = parent.joinpath(f"{results_dir}")
    if model1 == "gpt" or model1.startswith("jurassic"):
        two_affix1 = ""
    model1_path = parent_path1.joinpath(f"{model1}_{condition}{two_affix1}_prefix_{prefix1}.csv")

    is_random = False
    if model2.startswith("random"): 
        is_random = True
        model2_path=None
    else:
        if model2 == "gpt" or model2.startswith("jurassic"):
            two_affix2 = ""
        model2_path = parent_path2.joinpath(f"{model2}_{condition}{two_affix2}_prefix_{prefix2}.csv")

    return execute_mcnemar_agent_patient(model1, model1_path, model2, model2_path, is_random, condition)

def has_swap(exp):
    assert(type(exp.results[0]['swap_names']) == bool)
    return any([x['swap_names'] for x in exp.results[0:30]])

def filter_exps(exp1, exp2): 
    if has_swap(exp1) and not has_swap(exp2):
        exp1.results = [x for x in exp1.results if not x['swap_names']]
    if has_swap(exp2) and not has_swap(exp1):
        exp2.results = [x for x in exp2.results if not x['swap_names']]
    return exp1, exp2

def execute_mcnemar_general(model1, model1_path, model2, model2_path):
    exp1  = Experiment(model1, "", None, None, 1, None)
    exp1.recover(model1_path)
    exp2 = Experiment(model2, "", None, None, 1, None) 
    exp2.recover(model2_path)

    # if one has swap names and the other doesn't then filter the one that does down 
    exp1, exp2 = filter_exps(exp1, exp2)

    table = {model1: {"correct": [], "incorrect": []},
            model2: {"correct": [], "incorrect": []}}
        
    for i, (m1, m2) in enumerate(zip(exp1.results, exp2.results)):
        # drop other
        if m1['pred'].strip() == m1['true'].strip() and m1['pred'].strip() != 'other':
            table[model1]['correct'].append(i)
        elif m1['pred'].strip() != 'other': 
            # print("m1", m1['pred'], m1['true'])
            table[model1]['incorrect'].append(i)
        else:
            pass 
        if m2['pred'].strip() == m2['true'].strip() and m2['pred'].strip() != 'other':
            table[model2]['correct'].append(i)
        elif m2['pred'].strip() != 'other':
            # print("m2", m2['pred'], m2['true'])
            table[model2]['incorrect'].append(i)
        else:
            pass

    table_arr = np.zeros((2,2))
    corr1_corr2 = set(table[model1]['correct']) & set(table[model2]['correct'])
    incorr1_corr2 = set(table[model1]['incorrect']) & set(table[model2]['correct'])
    corr1_incorr2 = set(table[model1]['correct']) & set(table[model2]['incorrect'])
    incorr1_incorr2 = set(table[model1]['incorrect']) & set(table[model2]['incorrect'])

    table_arr[0,0] = len(corr1_corr2)
    table_arr[1,0] = len(incorr1_corr2)
    table_arr[0,1] = len(corr1_incorr2)
    table_arr[1,1] = len(incorr1_incorr2)

    mac = mcnemar(table_arr, exact=True, correction=False)
    return mac.pvalue, mac.statistic, table, table_arr