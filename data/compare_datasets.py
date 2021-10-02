# create plot summarising a single dataset/graph
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

def type_histogram(ax: Axes, graph:nx.Graph, nodes_requested):
    node_data = [graph.nodes[n] for n in nodes_requested]
    types = [n['class'] for n in node_data]
    ax.hist(types)  # TODO pick from qualitative color map
    ax.set_ylabel("number of aliases")
    ax.set_xlabel("node type")
    ax.tick_params(axis="x", labelrotation=60)
    # TODO labels centered below bars
    # TODO make this a horizontal bar chart
    return ax

def degree_histogram(ax: Axes, graph: nx.Graph, included):
    degrees = [deg for (node, deg) in graph.degree(included)]
    n, bins, patches = ax.hist(
        degrees,
        bins=max(degrees)
    )

    # TODO set xticks only for non-empty bars
    # or do not set at all but print in or over bar
    ax.set_xticks(range(max(degrees)))
    ax.set_xlabel("node degree")
    ax.set_ylabel("number of nodes")

    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for frac, patch in zip(fracs, patches):
        color = plt.cm.plasma(norm(frac))
        patch.set_facecolor(color)

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

    model = graph.graph['model']
    simple_nxG, proj_nxG = nx_get_interpretations(graph)

    n_species = len(model.species)
    n_reactions = len(model.reactions)

    # degrees of aliases (in simple graph)
    non_rxn_nodes = get_non_rxn_nodes(simple_nxG)
    n_nodes = len(non_rxn_nodes)  # number of aliases (nodes) in graph
    included, excluded = get_prediction_nodes(simple_nxG)
    included = np.intersect1d(included, non_rxn_nodes)
    n_nodes_predict = len(included)  # number of nodes considered for prediction

    # TODO like this no labels are set yet
    #   do this based on load_graphs

    pos_nodes = [node for (node, label) in graph.nodes(data="node_label", default=False) if label==1]
    n_pos = len(pos_nodes)

    fig, axs = init_fig(rows=3)

    ax_degs_all = degree_histogram(axs[0], simple_nxG, non_rxn_nodes)
    # TODO set sizes globally
    # maybe somehow like this
    # TODO decrease size of tick labels?
    ax_degs_all.xaxis.label.set_size(8)
    set_title_fancy(ax_degs_all, f"all species aliases")

    ax_degs_positive = degree_histogram(axs[1], simple_nxG, pos_nodes)
    set_title_fancy(ax_degs_positive, f"aliases w/ positive ground-truth class")

    ax_type_hist = type_histogram(axs[2], simple_nxG, non_rxn_nodes)
    # make clear that this does not count proteins that are contained in a complex species

    fig.suptitle(f"{graph.graph['name']}")
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    names = [
        "AlzPathwayReorgLast",
        # "PDMap19"
    ]

    graphs: list[deepsnap.graph.Graph] = load_graphs(
        names,
        loader_impl=sbml_single_bipartite_projection_impl
    )
    # TODO cache this

    for graph in graphs:
        fig = disease_map_summary(graph)
        fig.show()
