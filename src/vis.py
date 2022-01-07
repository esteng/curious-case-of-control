import seaborn as sns 
from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np 
import json 
import pathlib 
from experiment import Experiment

from metrics import accuracy_report

import re 
def barplot(csv_groups, level, x_name, hue_name, title=None, ax=None, filtered = False, ignore_first_only = False):
    # consolidate data from different csvs 
    all_dfs = {g:[] for g in csv_groups.keys()}
    for group, csvs in csv_groups.items():
        for csv in csvs:
            df = pd.read_csv(csv)
            # print(csv, len(df))
            all_dfs[group].append(df)

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

    g = sns.catplot(data = df_to_plot, kind='bar', x = x_name, y = 'acc', hue = hue_name, palette="colorblind", col="group")
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
    for csv, name in zip(csvs, names): 
        exp  = Experiment(name, exp_name, prompt, wrapper_fxn, 1, None)
        filename = pathlib.Path(csv).name
        exp.recover(csv)
        exp.recompute(nicknames, use_action=use_action, use_verb=use_verb, correct_idx=correct_idx)
        df = exp.format_results()
        df.to_csv(f"../{results_path}/{filename}")