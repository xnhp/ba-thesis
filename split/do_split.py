import numpy as np
import scipy
import sklearn.cluster
from graphgym.contrib.feature_augment.util import nx_get_interpretations
from graphgym.contrib.loader.SBML import sbml_single_bipartite_projection_impl
from graphgym.contrib.loader.util import load_nxG
from matplotlib import pyplot as plt
from run.visualisation import init_fig
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from data.models import get_dataset

def get_linkage_matrix(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    return linkage_matrix

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def foo():

    clustering = sklearn.cluster.AgglomerativeClustering(
        affinity="euclidean",
        # distance function, euclidean for layout info, may use semantic similarity for e.g. GO terms?
        linkage="single",  # makes most sense in our setup
        distance_threshold=None,  # ... above which clusters will not be merged — but want to compute entire dendrogram
        compute_distances=True,  # yes, please
        compute_full_tree=True
    ).fit(X)

    plt.scatter(X[:,0], X[:,1])
    plt.show()
    plot_dendrogram(clustering)
    plt.show()

    print("break")

    name = "AlzPathwayReorgLast"
    path, model_class = get_dataset(name)

    # load graph
    # assume training pipeline does not modify the graph
    # (not the case when considering subgraph of GO/BP annotations
    #   -- update if this ever becomes relevant)

    simple_nxG = load_nxG(path, model_class, name, collapsed=False)
    _, proj_nxG = nx_get_interpretations(simple_nxG)

    # TODO this will be aliases that have been predicted positive class
    # list of alias ids
    input_aliases: list[str] = ["sa5"]

    for alias_to_split in input_aliases:
        node = simple_nxG.nodes[alias_to_split]
        # as "neighbours" want to consider neighbour species, not reactions. can use bipartite proj for that.
        neighbs = [simple_nxG.nodes[n] for n in proj_nxG.neighbors(alias_to_split)]  # get node data dicts

        neighbs_cluster_feats = [[n['pos_x'], n['pos_y']] for n in neighbs]



        print("foo")


    print("hello")

    pass


def cluster(X):
    n_points = X.shape[0]
    if n_points <= 1:
        raise ValueError("Expecting at least degree 2")
    if n_points == 2:
        return None, None, None, 2, [0,1], None

    Z = linkage(X, method="single")
    dist_levels_rev = list(Z[::-1, 2])

    # use elbow method to determine number of clusters
    # or even smarter like here: https://www.scikit-yb.org/en/latest/api/cluster/elbow.html (.cf "knee point detection in python")
    # can also throw in this as a reference https://link.springer.com/article/10.1007/s00521-021-05873-3
    acceleration, step_sizes = None, None
    # add 0-dist step to allow case of all data points in their own cluster
    step_sizes = np.diff(dist_levels_rev + [0, 0], 1)
    # is negative because going downwards (descending order)
    n_clust = step_sizes.argmin() + 2

    if n_points >= 8:# need at least 8 points to have 2 steps of 2nd derivative
        acceleration = np.diff(dist_levels_rev, 2)
        n_clust = acceleration.argmax() + 2

    partition = fcluster(Z, n_clust, 'maxclust')
    return dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z


def plot_vis(dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z):
    fig, axs = init_fig(rows=3, cols=1)
    # TODO sizes of axes — make scatter and dend a bit taller
    ax_scatter = axs[0]
    ax_dend = axs[1]
    ax_line = axs[2]

    # TODO color mapping between dendrogram and scatterplot
    ax_scatter.scatter(X[:,0], X[:,1], c=partition)

    # TODO for line plot, instead of different colors, use different line styles.
    if dist_levels_rev is not None:
        ax_line.plot(range(1, len(Z)+2), dist_levels_rev + [0])  # aggregate differences of i..i+2
        if step_sizes is not None:
            ax_line.plot(range(2, len(Z)+3), step_sizes)  # aggregate diff of i..i+4
        if acceleration is not None:
            ax_line.plot(range(2, len(Z)), acceleration)  # 2nd deriv
        ax_line.legend(['dist level', 'step size (1st deriv)', 'acceleration (2nd deriv)'],
                       bbox_to_anchor=(0,1.02,1,0.2), mode="expand", ncol=3, loc="lower left")
        ax_line.set_xticks(range(len(Z)+2))
        ax_line.set_xlabel("number of clusters")
        ax_line.hlines([0], xmin=1, xmax=len(Z)+2, linestyle='--')

        ax_line.axvline(n_clust, color="r", linewidth=1, linestyle="--")
        # move line below merge point for more visual clarity
        offset = 0.5 * (dist_levels_rev[n_clust-2] - dist_levels_rev[n_clust] - 1)
        cut_thresh = dist_levels_rev[n_clust-2] - offset
        ax_dend.axhline(cut_thresh, color="r", linewidth=1, linestyle="--")
        # TODO animate cut line with camera

        dendrogram(Z, ax=ax_dend, color_threshold=cut_thresh, no_labels=True)


    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    # DONE plot degree distribution of nodes to consider
    # → def need to consider both cases (<= 4, and larger)

    # degree 1 is excluded
    # degree 2: split always
    # degree 3: fall back to 1st deriv (i.e. max distance gap)
    # degree >= 4: use

    Xs = [
        # np.array([[0,0], [10, 10]]),
        # np.array([[0, 0], [10, 10], [20, 22]]),
        # np.array([[4, 0.5], [10, 10], [10, 11]]),
        # np.array([[4, 0.5], [10, 10], [10, 11], [15, 15]]),
        # np.array([[4, 0.5], [10, 10], [20, 21], [15, 17]]),
        np.array([[1, 2.25], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0.5], [10, 10], [10, 11], [15, 15]])
    ]

    for X in Xs:
        dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z = cluster(X)
        plot_vis(dist_levels_rev, step_sizes, acceleration, n_clust, partition, Z)


    print("break")

























