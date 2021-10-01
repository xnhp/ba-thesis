import numpy as np
from graphgym.contrib.feature_augment.util import nx_get_interpretations, get_non_rxn_nodes, get_prediction_nodes
from graphgym.contrib.loader.SBML import sbml_single_bipartite_projection_impl
from graphgym.contrib.loader.gg_loaders import load_graphs
from run.visualisation import init_fig
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import deepsnap


def cluster(X:np.array):
    """
    :param X: node features to cluster (e.g. layout positions)
    """
    # degree 1: excluded
    # degree 2: split always
    # above, use elbow method to determine number of clusters
    # or even smarter like here: https://www.scikit-yb.org/en/latest/api/cluster/elbow.html (.cf "knee point detection in python")
    # can also throw in this as a reference https://link.springer.com/article/10.1007/s00521-021-05873-3
    # degree 3: fall back to 1st derivative (i.e. greatest distance gap)
    # degree >= 4: use 2nd derivative (greatest acceleration)

    n_points = X.shape[0]
    if n_points <= 1:
        raise ValueError("Expecting at least degree 2")
    if n_points == 2:
        return None, None, None, 2, [0, 1], None

    Z = linkage(X, method="single")
    dist_levels_rev = list(Z[::-1, 2])

    acceleration, step_sizes = None, None
    # add 0-dist step to allow case of all data points in their own cluster
    step_sizes = np.diff(dist_levels_rev + [0, 0], 1)
    # is negative because going downwards (descending order)
    n_clust = step_sizes.argmin() + 2
    # TODO: if two successive knee values always the same, prefer the latter (i.e. higher number of clusters)
    # ↝ 8b0957

    if n_points >= 8:  # need at least 8 points to have 2 steps of 2nd derivative
        acceleration = np.diff(dist_levels_rev, 2)
        n_clust = acceleration.argmax() + 2

    cut_thresh = dist_levels_rev[n_clust - 2]
    # TODO maxclust still may give fewer than n_clust clusters
    # partition = fcluster(Z, n_clust, 'maxclust')
    # hacky workaround
    eps = 0.01
    partition = fcluster(Z, t=cut_thresh-eps, criterion='distance')
    return dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z


def plot_vis(dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z, X):
    fig, axs = init_fig(rows=3, cols=1)
    # TODO sizes of axes — make scatter and dend a bit taller
    ax_scatter = axs[0]
    ax_dend = axs[1]
    ax_line = axs[2]

    # TODO color mapping between dendrogram and scatterplot
    ax_scatter.scatter(X[:, 0], X[:, 1], c=partition)
    # TODO just make them really small instead? they're not really relevant but probably bad practise not to give them
    #   also add x and y labels (also really small)
    ax_scatter.set_yticks([])
    ax_scatter.set_xticks([])

    # TODO for line plot, instead of different colors, use different line styles.
    if dist_levels_rev is not None and len(X)>2:
        ax_line.plot(range(1, len(Z) + 2), dist_levels_rev + [0])  # aggregate differences of i..i+2
        if step_sizes is not None:
            ax_line.plot(range(2, len(Z) + 3), step_sizes)  # aggregate diff of i..i+4
        if acceleration is not None:
            ax_line.plot(range(2, len(Z)), acceleration)  # 2nd deriv
        ax_line.legend(['dist level', 'step size (1st deriv)', 'acceleration (2nd deriv)'],
                       bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", ncol=3, loc="lower left")
        ax_line.set_xticks(range(len(Z) + 2))
        ax_line.set_xlabel("number of clusters")
        ax_line.set_yticks([])  # none
        ax_line.hlines([0], xmin=1, xmax=len(Z) + 2, linestyle='--')

        # illustrate cut line
        ax_line.axvline(n_clust, color="r", linewidth=1, linestyle="--")
        # move line below merge point for more visual clarity
        # only do this a little because it must be above any lower merge
        offset = 0.05 * (dist_levels_rev[n_clust-2] - dist_levels_rev[n_clust-1])
        cut_thresh = dist_levels_rev[n_clust - 2]
        ax_dend.axhline(cut_thresh - offset, color="r", linewidth=1, linestyle="--")
        # TODO animate cut line with camera

        dendrogram(Z, ax=ax_dend, color_threshold=cut_thresh, no_labels=True)
        ax_dend.set_yticks([])

    return fig


def memberships_to_partition(memberships, aliases):
    partition = {}
    for cluster_id, alias_id in zip(memberships, aliases):
        if cluster_id not in partition:
            partition[cluster_id] = []
        partition[cluster_id].append(alias_id)
    return partition


if __name__ == '__main__':

    # DONE plot degree distribution of nodes to consider
    # → def need to consider both cases (<= 4, and larger)

    # some dummy input data
    # Xs = [
    #     np.array([[0,0], [10, 10]]),
    #     np.array([[0, 0], [10, 10], [20, 22]]),
    #     np.array([[4, 0.5], [10, 10], [10, 11]]),
    #     np.array([[4, 0.5], [10, 10], [10, 11], [15, 15]]),
    #     np.array([[4, 0.5], [10, 10], [20, 21], [15, 17]]),
    #     np.array([[1, 2.25], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0.5], [10, 10], [10, 11], [15, 15]])
    # ]
    #
    # for X in Xs:
    #     dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z = cluster(X)
    #     plot_vis(dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z, X)

    names = [
        # "AlzPathwayReorgLast",
        "PDMap19"
    ]

    graphs: list[deepsnap.graph.Graph] = load_graphs(
        names,
        loader_impl=sbml_single_bipartite_projection_impl
    )
    # important to use bipartite proj here because we will use that to determine neighbours
    # TODO what if reaction has degree > 2 and other node is somewhere else altogether?
    # TODO reason why it makes sense to consider bipartite proj

    for graph in graphs:
        simple_nxG, proj_nxG = nx_get_interpretations(graph)
        non_rxn_nodes = get_non_rxn_nodes(simple_nxG)
        n_nodes = len(non_rxn_nodes)  # number of aliases (nodes) in graph
        included, excluded = get_prediction_nodes(simple_nxG)
        included = np.intersect1d(included, non_rxn_nodes)
        pos_nodes = [node for (node, label) in graph.nodes(data="node_label", default=False) if label == 1]
        pos_nodes = pos_nodes[4:16]
        # pos_nodes = ["sa10250"]

        for alias_to_split in pos_nodes:  # TODO first few for debugging
            node = simple_nxG.nodes[alias_to_split]
            # as "neighbours" want to consider neighbour species, not reactions. can use bipartite proj for that.
            neighbs = [simple_nxG.nodes[n] for n in proj_nxG.neighbors(alias_to_split)]  # get node data dicts
            neighbs_cluster_feats = [[n['pos_x'], n['pos_y']] for n in neighbs]
            X = np.array(neighbs_cluster_feats)
            dist_levels_rev, step_sizes, acceleration, n_clust, memberships, Z = cluster(X)

            partition = memberships_to_partition(memberships, neighbs)

            fig = plot_vis(dist_levels_rev, step_sizes, acceleration, n_clust, memberships, Z, X)
            fig.suptitle(f"{alias_to_split} (deg {len(neighbs)}) \n in {graph.graph['name']}")
            fig.tight_layout()
            fig.show()


