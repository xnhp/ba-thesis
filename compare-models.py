# directory containing dirs for each run (commonly called "config-{hyperparms}"
import csv
import os

import numpy as np
import pandas as pd
from graphgym.utils.io import json_to_dict_list
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics

from data.summary import print_model_summary, print_graph_summary

# experiment_name = "train-on-many"
# config_name = "config"
experiment_name = "AD-reorg-playground"
config_name = "config-gnn"

experiment_dir = os.path.join("GraphGym/run/", experiment_name)
grid_out_dir = os.path.join(experiment_dir, "generated-configs/", config_name + "_grid_grid")
plot_out_dir = os.path.join(experiment_dir, "results")



def get_model_details(model_dir):
    """
    :param model_dir: directory containing results for repeats (dirs with numbers as names) and "agg" that contains
        results aggregated across repeats
    :return: dict
    """

    def find_stats(key):
        # rely on GG aggregation across repeats and only consider "agg" directory
        # (triggered at end of main.py)
        agg_path = os.path.join(model_dir, 'agg', key)
        best_path =  os.path.join(agg_path, "best.json")
        stats_path =  os.path.join(agg_path, "stats.json")

        if os.path.exists(best_path):
            # in case of SVM, stats.json contains a line for each eval epoch,
            # and additionally, best.json is created, selecting the best line according to
            # a configured metric
            return json_to_dict_list(best_path)[0]
        else:
            # in case of SVM we only print stats.json, which contains a single line (only one "epoch")
            # -- nothing to aggregate
            stats = json_to_dict_list(stats_path)
            assert len(stats) == 1
            return stats

    return {
        'name': os.path.basename(model_dir),
        'path': model_dir
    } | {
        key: find_stats(key)
        for key in ['train', 'val', 'val-graph']
    }


def read_model_results(out_dir):
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
    return list(zip(s, v))


def top_k_on_split(models, split, metric):
    k = 5
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

def roc_thresh(model, split):
    y_true, y_proba = get_prediction_and_truth(model, split)
    return _roc_thresh(y_true, y_proba)

def _roc_thresh(y_true, y_proba):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr, tpr, optimal_threshold, optimal_idx

def save_loss(model, split):
    try:
        targetpath = os.path.join(model['path'], "1", split, "stats.json")
    except FileNotFoundError:
        return
    dictlist = json_to_dict_list(targetpath)
    plt.figure(figsize=(6, 3), facecolor='lightgray')
    plt.title(f"Loss ({split})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot([epoch['loss'] for epoch in dictlist])
    plt.savefig(os.path.join(plot_out_dir, "loss_"+split))
    plt.close()

def save_roc(model, split="train", plot_random=True):
    fpr, tpr, _, _ = roc_thresh(model, split)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(f'Receiver Operating Characteristic ({split})')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    if plot_random:
        y_true, _ = get_prediction_and_truth(model, split)
        y_score = np.random.random(len(y_true))
        fpr, tpr, _, _ = _roc_thresh(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'r', label='AUC (random cl.) = %0.2f' % roc_auc)
    
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ticks = np.append(np.arange(0,1,step=0.25), 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(b=True, linestyle="dotted")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(plot_out_dir, "roc_" + split))
    plt.close()

def get_filename(split):
    # ouch.
    if split == "train":
        return "train"
    elif split == "val":
        return "test"
    elif split == "val-graph":
        return "val"

def get_prediction_and_truth(model, split):
    svm_pred_path = os.path.join(model['path'], "1", split)
    # gnn_pred_base_path = os.path.join(model['path'], "1", split, "preds")
    if os.path.exists(os.path.join(svm_pred_path, "Y_" + get_filename(split) + ".csv")):
        y_test = read_pd_csv(os.path.join(svm_pred_path, "Y_" + get_filename(split) + ".csv"))
        # only probs of positive class
        probas = read_pd_csv(os.path.join(svm_pred_path, "pred_" + get_filename(split) + ".csv"))
        if probas.shape[1] == 2:  # in case we forgot to limit to column of positive class when writing
            probas = pd.DataFrame(probas['1'])
        return y_test, probas
    else:  # in case of gnn we write preds for each eval epoch ↝ train.py
        # eval_epoch_dirs = [dir for dir in os.scandir(gnn_pred_base_path)]
        # get preds of best epoch
        # find best performance on external validation split
        best = json_to_dict_list(os.path.join(model['path'], 'agg', 'val-graph', 'best.json'))
        best_epoch_ix = best[0]['epoch']
        best_pred_dir = os.path.join(model['path'], '1', split, 'preds', str(best_epoch_ix))
        y_test = read_pd_csv(os.path.join(best_pred_dir, "Y_" + get_filename(split) + ".csv"))
        probas = read_pd_csv(os.path.join(best_pred_dir, "pred_" + get_filename(split) + ".csv"))
        return y_test, probas


def read_identifier_mapping(model):
    # cleanup
    dir = os.path.join(model['path'], '1', 'val-graph')
    id_mapping_path = os.path.join(dir, 'mapping_int_to_alias.csv')
    identifier_mapping = {}
    with open(id_mapping_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        for row in reader:
            if len(row) == 2:
                identifier_mapping.update({int(row[0]): row[1]})
        # data_read = [row for row in reader]

    node_label_index_path = os.path.join(dir, 'node_label_index.csv')
    node_label_index = read_pd_csv(node_label_index_path)

    return identifier_mapping, node_label_index


def zip_safe(a, b, c):
    assert len(a) == len(b) == len(c)
    return zip(a, b, c)


def conf_mat_aliases(model):
    """
    Returns aliases of true/false positives/negatives in shape confusion matrix
    :param model:
    :return:
    """
    class_preds, y_proba, y_true = get_class_preds(model, 'val-graph')
    identifier_mapping, node_label_index = read_identifier_mapping(model)
    y_proba = list(y_proba.get('0'))
    y_true = list(y_true.get('0'))
    node_label_index = list(node_label_index.get('0'))
    ids_for_nodes = [identifier_mapping[ix] for ix in node_label_index]
    # these should correspond
    # ok nice so far, but also need node_label_index
    preds_true_alias = list(zip_safe(class_preds, y_true, ids_for_nodes))

    # assemble sth like a confusion matrix
    tns = [alias for pred, true, alias in preds_true_alias if pred == 0 and true == 0]
    fns = [alias for pred, true, alias in preds_true_alias if pred == 0 and true == 1]
    fps = [alias for pred, true, alias in preds_true_alias if pred == 1 and true == 0]
    tps = [alias for pred, true, alias in preds_true_alias if pred == 1 and true == 1]
    print(
        len(tps), len(tns), len(fps), len(fns)
    )

    return [[tns, fns], [fps, tps]]


def save_conf_mat(model, tresh=None, split="train"):
    y_true, y_proba = get_prediction_and_truth(model, split)

    if tresh is None:
        _, _, tresh, _ = roc_thresh(model, split)

    def decision_function(prob):
        return 0 if float(prob) < tresh else 1

    class_preds = [decision_function(prob) for prob in y_proba.values]
    conf_mat = metrics.confusion_matrix(y_true, class_preds)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mat)
    plt.title(f"Confusion Matrix (t={tresh} on {split})")
    plt.savefig(os.path.join(plot_out_dir, "confusion_" + split))
    # TODO instead, print identifiers of datasets in splits?
    plt.close()


def get_class_preds(model, split, thresh=None):
    """
    Turn soft classifier confidence values into crisp class predictions based on threshold. If no threshold given,
    determine it automatically based on maximum margin between TPR and FPR
    :return:
    """
    y_true, y_proba = get_prediction_and_truth(model, split)
    if thresh is None or thresh == "auto":
        _, _, tresh, _ = roc_thresh(model, split)

    def decision_function(prob):
        return 0 if float(prob) < tresh else 1

    class_preds = [decision_function(prob) for prob in y_proba.values]
    return class_preds, y_proba, y_true


def read_yaml(path):
    import yaml
    with open(path) as f:
        dataMap = yaml.safe_load(f)
    return dataMap


def map_summary(identifier):
    model, s1 = print_model_summary(identifier)
    graph, s2 = print_graph_summary(model)
    return s1 + s2


def get_used_datasets(mdls):
    # obtain identifiers of used maps
    dummy_mdl = mdls[0]
    dummy_config = read_yaml(os.path.join(dummy_mdl['path'], "1", "config.yaml"))
    train_identifiers = dummy_config['dataset']['train_names']
    # i.e. corresponding to val-graph
    test_identifiers = dummy_config['dataset']['test_names']
    return train_identifiers, test_identifiers

def data_summary(models):
    s = ""
    train_ids, test_ids = get_used_datasets(models)
    s += "Models used for training (internal):\n"
    for train_id in train_ids:
        # s += map_summary(train_id)
        s += "\n"
    s += "Models used for validation (external):\n"
    for test_id in test_ids:
        # s += map_summary(test_id)
        s += "\n"
    return s

def tpr_cutoffs(model, split, cutoffs=None):
    if cutoffs is None:
        cutoffs = [0.25, 0.5, 0.75]
    fpr, tpr, t_opt, t_opt_ix = roc_thresh(model, split)
    return {
        # find fpr at (close) a given tpr
        # i.e. "if we want to receive {tpr}% of true positives, how many false positives do we get?"
        tpr_cutoff: fpr[
            # find index of largest tpr <= cutoff
            # assume tpr is in ascending order!
            len([r for r in tpr if r <= tpr_cutoff]) - 1
        ]
        for tpr_cutoff in cutoffs
    }

def tpr_cutoffs_str(model, split):
    s = ""
    cutoffs = tpr_cutoffs(model, split)
    s += "FPR at TPR cutoffs: \n"
    for tpr_cutoff, fpr in cutoffs.items():
        s += f"\t{tpr_cutoff:.2f}:\t {fpr:.3f}\n"
    return s



if __name__ == "__main__":

    models = read_model_results(grid_out_dir)

    fav_mdl, _ = sort_models("val-graph", 'auc', models)[0]

    config_mdl = get_model_details(os.path.join(experiment_dir, "results", config_name))

    # model_to_inspect = config_mdl
    model_to_inspect = fav_mdl

    for split in ["train", "val", "val-graph"]:
        # TODO arrange these in subplots
        save_roc(model_to_inspect, split=split)
        save_conf_mat(model_to_inspect, split=split)
        save_loss(model_to_inspect, split=split)

    with open(os.path.join(plot_out_dir, "summary.txt"), "w") as f:
        # TODO info about each split
        f.write(data_summary(models))
        f.write(top_k_all_splits(models, 'auc'))
        f.write(top_k_all_splits(models, 'accuracy'))
        f.write(f"info on fav mdl: {model_to_inspect['name']}\n")
        f.write(str(tpr_cutoffs_str(model_to_inspect, 'val-graph')))
        f.write("\n (see folder for plots)")



