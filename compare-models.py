# directory containing dirs for each run (commonly called "config-{hyperparms}"
import os

import pandas as pd
from graphgym.utils.io import json_to_dict_list
from matplotlib import pyplot as plt
from sklearn import metrics

from data.util import print_model_summary, print_graph_summary

from graphgym.config import cfg

out_dir = "GraphGym/run/sanity/generated-configs/config_grid_grid"


def get_model_details(model_dir):
    """
    :param model_dir: directory containing results for repeats (dirs with numbers as names) and "agg" that contains
        results aggregated across repeats
    :return: dict
    """

    def find_stats(key):
        return json_to_dict_list(
            # rely on GG aggregation across repeats and only consider "agg" directory
            # (triggered at end of main.py)
            # Read from stats.json which contains averages and stddev ↝ GraphGym/graphgym/utils/agg_runs.py:48
            os.path.join(model_dir, "agg", key, "stats.json")
        )[0]  # stats.json will always contain only a single line

    d = {
        'name': os.path.basename(model_dir),
        'path': model_dir
    }
    d.update({
        key: find_stats(key)
        for key in ['train', 'val', 'val-graph']
    })
    return d


def flobble(out_dir):
    model_dirs = [dir for dir in os.scandir(out_dir) if dir.is_dir()]
    # dict from model name (arbitrary, lets make this the name of the directory)
    # to additional information such as performance metrics or predictions
    return [
        get_model_details(model_dir) for model_dir in model_dirs
    ]


def sort_models(split, key, model_details):
    key_func = lambda mdl: mdl[split][key]
    s = sorted(model_details,
               key=key_func,
               reverse=True)
    # return extracted values for convenience
    v = [key_func(mdl) for mdl in s]
    return list(zip(s,v))


def top_k_on_split(models, split, metric):
    k=3
    s1 = sort_models(split, metric, models)[:k]
    s = ""
    split_desc = {
        'train': "internal train split",
        'val': "internal test/validate split",
        'val-graph': "external test/validate split"
    }
    s += ("top " + str(k) + " models on " + split_desc[split] + " by " + metric + "\n")
    for model, value in s1:
        s += ("\t" + model['name'] + ":\t " + str(value) + "\n")
    return s

def top_k_all_splits(models, metric):
    s = ""
    for split in ['train', 'val', 'val-graph']:
        s += top_k_on_split(models, split, metric)
    s += ("\n")
    return s


def split_info(mdls, split):
    # TODO: switch to name the csv files the same regardless of the split they are in
    #       so we dont have to switch cases here
    # dummy_mdl = mdls[0]
    # y_test = read_pd_csv(os.path.join(dummy_mdl['path'], "1", "val-graph", "Y_val.csv"))
    # print("foo")
    pass



def read_pd_csv(path):
    df = pd.read_csv(path)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df

def save_roc(model):
    y_test = read_pd_csv(os.path.join(model['path'], "1", "val-graph", "Y_val.csv"))
    # only probs of positive class
    probas = read_pd_csv(os.path.join(model['path'], "1", "val-graph", "pred_val.csv"))['1']
    fpr, tpr, threshold = metrics.roc_curve(y_test, probas)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(out_dir, "roc"))


def save_conf_mat(model, tresh):
    # TODO https://vitalflux.com/python-draw-confusion-matrix-matplotlib/
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    #   https://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/
    pass


if __name__ == "__main__":

    # TODO print network summaries of used networks
    # dont have cfg initialised here, would either need to call this
    # via python in main.py (no) or read from yaml or hardcode
    # for name in cfg.dataset.train_names:
    #     sbmlmdl = print_model_summary(name)
    #     graph = print_graph_summary(name)
    # for name in cfg.dataset.test_names:
    #     sbmlmdl = print_model_summary(name)
    #     graph = print_graph_summary(name)

    # improve this by using logging module? https://stackoverflow.com/a/9321890/156884

    models = flobble(out_dir)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(top_k_all_splits(models, 'auc'))
        f.write(top_k_all_splits(models, 'accuracy'))

    fav_mdl, _ = sort_models("val-graph", 'auc', models)[0]
    save_roc(fav_mdl)

    split_info(models, "val")

    # save_conf_mat(fav_mdl, t_opt)
