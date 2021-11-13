# create plot summarising a single dataset/graph
import os

import pylab
from graphgym.contrib.loader.SBML import sbml_single_bipartite_projection_impl
from graphgym.contrib.loader.gg_loaders import load_graphs

import deepsnap.graph
import networkx as nx
import numpy as np
from graphgym.contrib.feature_augment.util import nx_get_interpretations, get_non_rxn_nodes, get_prediction_nodes, \
    split_by_predicate
from graphgym.contrib.loader.util import load_nxG
from matplotlib import colors, pyplot as plt
from matplotlib.axes import Axes
from run.visualisation import init_fig

from data.models import get_dataset


def set_title_fancy(ax, title: str):
    fontdict = {
        'fontsize': 8
    }
    ax.set_title(title, fontdict=fontdict)


def info_text(ax, graph: nx.Graph):
    model = graph.graph['model']
    simple_nxG, proj_nxG = nx_get_interpretations(graph)
    n_species = len(model.species)
    n_reactions = len(model.reactions)

    # degrees of aliases (in simple graph)
    non_rxn_nodes = get_non_rxn_nodes(simple_nxG)
    n_nodes = len(non_rxn_nodes)  # number of aliases (nodes) in graph
    assert n_species == n_nodes  # should be same in collapsed graph

    included, excluded = get_prediction_nodes(simple_nxG)
    included = np.intersect1d(included, non_rxn_nodes)
    n_nodes_predict = len(included)  # number of nodes considered for prediction

    pos_nodes = [node for (node, label) in graph.nodes(data="node_label", default=False) if label == 1]
    n_pos = len(pos_nodes)

    # hide axis, border, ...
    ax.axis('off')

    # this is still bottom-aligned but grows to the top... see if we need that
    textparams = {
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'transform': ax.transAxes
    }
    ax.text(0, 0,
            f"{n_nodes} species aliases, {n_reactions} reactions\n"
            f"{n_nodes_predict} ({(n_nodes_predict/n_nodes)*100:.2f}%) for prediction\n"
            f"{n_pos} ({(n_pos / n_nodes_predict)*100:.2f}%) to-be-duplicated",
            **textparams
            )
    return ax


def type_histogram(ax: Axes, graph: nx.Graph, nodes_requested):
    node_data = [graph.nodes[n] for n in nodes_requested]
    types = [n['class'] for n in node_data]
    ax.hist(types)  # TODO pick from qualitative color map
    ax.set_ylabel("number of aliases")
    ax.set_xlabel("node type")
    ax.tick_params(axis="x", labelrotation=60)
    # TODO labels centered below bars
    # TODO make this a horizontal bar chart
    return ax


def degree_histogram(ax: Axes, graph: nx.Graph):
    model = graph.graph['model']
    simple_nxG, proj_nxG = nx_get_interpretations(graph)
    n_species = len(model.species)
    n_reactions = len(model.reactions)

    # degrees of aliases (in simple graph)
    non_rxn_nodes = get_non_rxn_nodes(simple_nxG)
    n_nodes = len(non_rxn_nodes)  # number of aliases (nodes) in graph
    assert n_species == n_nodes  # should be same in collapsed graph

    included, excluded = get_prediction_nodes(simple_nxG)
    included = np.intersect1d(included, non_rxn_nodes)
    n_nodes_predict = len(included)  # number of nodes considered for prediction

    max_degree = 500
    degrees = [deg for (node, deg) in graph.degree(included) ]
    allowed_degrees = [deg for deg in degrees if deg <= max_degree]
    print(f"degree outliers: {[d for d in degrees if d > max_degree]}")
    n, bins, patches = ax.hist(
        allowed_degrees,
        bins=60
    )

    # TODO set xticks only for non-empty bars
    # or do not set at all but print in or over bar

    fontdict = {
        'fontsize': 8
    }
    # ax.set_xticks(range(max(degrees)))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("node degree", fontdict=fontdict)
    ax.set_ylabel("number of nodes (log)", fontdict=fontdict)
    ax.set_yscale('log')

    # fracs = n / n.max()
    # norm = colors.Normalize(fracs.min(), fracs.max())
    # for frac, patch in zip(fracs, patches):
    #     color = plt.cm.plasma(norm(frac))
    #     patch.set_facecolor(color)

    return ax


def single_bar(ax, graph):
    labels = [1]
    class_counts = [4,9,15]
    class_labels = ['foo', 'bar', 'baz']
    # ax.barh([1], [4], label="foo")
    # ax.barh([1], [5], label="foo", left=4)

    simple_nxG, proj_nxG = nx_get_interpretations(graph)
    # degrees of aliases (in simple graph)
    non_rxn_nodes = get_non_rxn_nodes(simple_nxG)
    node_data = [graph.nodes[n] for n in non_rxn_nodes]
    types = [n['class'] for n in node_data]

    from collections import Counter
    type_counts = Counter(types)

    expected_types = [
        "PROTEIN",
        "RNA",
        "DEGRADED",
        "DRUG",
        "SIMPLE_MOLECULE",
        "PHENOTYPE",
        "GENE",
        "ION",
        "UNKNOWN",
        "COMPLEX"
    ]

    # TODO ordering persistent across networks
    # TODO pad with 0 if type does not appear

    # for ((curr_label, curr_count), prev_count) in zip(type_counts.items(), [0] + list(type_counts.values())):
    #     ax.barh([1], [curr_count], left=prev_count, label=curr_label)

    # pad with 0 and sort by key s.t. we have persistent row order across plots
    for expected_type in expected_types:
        if expected_type not in type_counts:
            type_counts[expected_type] = 0
    type_counts = {k: v for k, v in sorted(type_counts.items(), key=lambda item: item[0])}

    fontdict = {
        'fontsize': 8
    }
    # ax.set_xticks(range(max(degrees)))
    ax.tick_params(labelsize=8)
    y_pos = np.arange(len(type_counts))
    ax.barh(y_pos, type_counts.values())
    ax.set_yticks(y_pos)
    ax.set_yticklabels(type_counts.keys())
    ax.invert_yaxis()
    ax.set_xlabel("Number of species", fontdict=fontdict)

    set_title_fancy(ax, "Species type distribution")
    # TODO combined legend for all networks
    # ax.legend(
    #     bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", ncol=3, loc="lower right"
    # )
    # ax.legend()
    # ax.axis('off')
    # for count, desc in zip(class_counts, class_labels):
    #     ax.bar(labels, [count], label=desc)
    return ax


def disease_map_summary(graph: nx.Graph):
    # path, model_class = get_dataset(name)
    # model = model_class(path)
    # construct graph
    # assume training pipeline does not modify the graph
    # (not the case when considering subgraph of GO/BP annotations
    #   -- update if this ever becomes relevant)
    # simple_nxG = load_nxG(path, model_class, name, collapsed=False, model=model)
    # _, proj_nxG = nx_get_interpretations(simple_nxG)

    # fig, axs = init_fig(rows=3)
    fig = plt.figure(constrained_layout=True, figsize=(5,6))
    fig.set_dpi(140)
    gs = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[0.3,2.25,2.3])

    ax_text = fig.add_subplot(gs[0])
    ax_text = info_text(ax_text, graph)

    ax_bar = fig.add_subplot(gs[1])
    ax_bar = single_bar(ax_bar, graph)

    ax_degs_all = fig.add_subplot(gs[2])
    ax_degs_all = degree_histogram(ax_degs_all, graph)
    set_title_fancy(ax_degs_all, "Degree distribution of species alias nodes")

    # # TODO set sizes globally
    # # maybe somehow like this
    # # TODO decrease size of tick labels?
    # ax_degs_all.xaxis.label.set_size(8)
    # set_title_fancy(ax_degs_all, f"all species aliases")
    #
    # ax_degs_positive = degree_histogram(axs[1], simple_nxG, pos_nodes)
    # set_title_fancy(ax_degs_positive, f"aliases w/ positive ground-truth class")
    #
    # ax_type_hist = type_histogram(axs[2], simple_nxG, non_rxn_nodes)
    # # make clear that this does not count proteins that are contained in a complex species

    fig.suptitle(f"{graph.graph['name']}")
    # fig.tight_layout()
    fig.tight_layout(h_pad=1.2)

    return fig


if __name__ == '__main__':
    names = [
        # "NF-kB"
        # "AlzPathwayReorgLast",
        # "PDMap19",
        "ReconMapOlder"
    ]

    graphs: list[deepsnap.graph.Graph] = load_graphs(
        names,
        loader_impl=sbml_single_bipartite_projection_impl
    )
    # TODO cache this

    for graph, name in zip(graphs, names):
        fig = disease_map_summary(graph)
        fig.show()
        fig.savefig(
            os.path.join(
            "C:\\Users\\Ben\\Uni\\BA-Thesis\\written\\images\\generated",
            name + ".png"
            )
        )
