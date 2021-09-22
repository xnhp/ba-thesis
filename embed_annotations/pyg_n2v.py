# adapted from https://github.com/rusty1s/pytorch_geometric/
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/node2vec.py
import os
from importlib.resources import files

import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import Node2Vec


# if invoking this from console (not IDE), you may need to
# set PYTHONPATH before, like in some `run.sh` file.
# This worked:
# export PYTHONPATH="${PYTHONPATH}/c/Users/Ben/Uni/BA-Thesis"
# export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/Uni/BA-Thesis/GraphGym/graphgym"
# export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/Uni/BA-Thesis/GraphGym/run"
# export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/git-repos/deepsnap"

# Also seems like even though we configure cuda, running this out of PyCharm
#   does not use GPU (at least running from console is much faster)


def _from_networkx(G):
    # copy of from_networkx except that it does not convert node labels to ints (doing that ourselves)
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


def embed_GO_pyG(nxGO):
    # node ids are GO identifiers (strings)
    nxGO = nx.convert_node_labels_to_integers(nxGO, ordering="sorted", label_attribute="go_term")
    print("converting to PyTorch Geometric dataset")
    data = _from_networkx(nxGO)  # 28.748 nodes, 106.030 edges
    # ↑ will contain data['go_term']: list of terms. guess we can assume these will map to ids.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=80,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=0.5, sparse=True).to(device)
    # ↝ [[^e16bdc]]

    # set num workers to 0 because pytorch multiprocessors does not work on windows
    # ↝ https://github.com/fastai/fastbook/issues/85#issuecomment-614000930
    # seems to work
    # set batch size 128 → 64 → 32 because getting out-of-memory errors
    loader = model.loader(batch_size=32, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # @torch.no_grad()
    # def test():
    #     model.eval()
    #     z = model()
    #     acc = model.test(z[data.train_mask], data.y[data.train_mask],
    #                      z[data.test_mask], data.y[data.test_mask],
    #                      max_iter=150)
    #     return acc

    print("starting train loop")
    # seems to converge to 0.75 after ~85 its
    # with updated hyperparams seemed to converge much faster
    # 0.73 at epoch 06 seems to be best, only gets slightly worse after that
    for epoch in range(1, 7):
        loss = train()
        # acc = test()
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    z = model(torch.arange(data.num_nodes, device=device))
    z = z.detach()
    # data['go_term'] already is a normal list and will also be read as such?
    return z.cpu().numpy(), data['go_term']



    # takes >1 min for larger datasets (e.g. GO/BP)
    # @torch.no_grad()
    # def plot_points(colors):
    #     print("plotting points")
    #     model.eval()
    #     # make a prediction for all nodes
    #     z = model(torch.arange(data.num_nodes, device=device))
    #     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    #     # y = data.y.cpu().numpy()
    #
    #     plt.figure(figsize=(8, 8))
    #     plt.scatter(z[0], z[1], s=5)
    #     # for i in range(dataset.num_classes):
    #     #     plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    #     plt.axis('off')
    #     plt.show()
    #
    # colors = [
    #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700'
    # ]
    # plot_points(colors)


cache_dir = files('computed')
def load_GO_graph():
    graph_file = os.path.join(cache_dir, "GeneOntology", "ontology-graph.gml")
    if os.path.exists(graph_file):
        # load picked (serialised) networkx graph. This is faster than parsing the .obo ontology file
        # and constructing the networkx graph each time.
        return nx.read_gml(graph_file)
        print("read ontology graph")

    path = os.path.join(files("data"), "GeneOntology", "go-basic.json.gz")  # load from data dir (not cache)
    # can also load from remote resource
    url = "http://release.geneontology.org/2021-02-01/ontology/go-basic.json.gz"
    # TODO can we omit certain relations/edges? do we want to?
    # ruiz et al consider only regulates, positively regulates, negatively regulates, part of, and is a
    # but looking at a handful examples of this case shows that these all only have "involved in"
    nxo = from_file(path)
    # GO:0008150 is biological process root term/node
    components = [nxo.graph.subgraph(c) for c in nx.weakly_connected_components(nxo.graph)]
    bp_comp = [comp for comp in components if "GO:0008150" in comp][0]
    bp_comp = bp_comp.to_undirected(reciprocal=False, as_view=True)
    nx.write_gml(bp_comp, graph_file)
    print("wrote ontology graph")
    return bp_comp

def embed_GO(force_recompute=False):
    nxGO = load_GO_graph()
    embedding_filename = os.path.join(cache_dir, "GeneOntology", "embedding.pt")
    term_filename = os.path.join(cache_dir, "GeneOntology", "term_ids.pt")
    if os.path.exists(embedding_filename) and not force_recompute:  # assume term_filename exists aswell
        print("loading computed embedding from disk")
        embs, terms = torch.load(embedding_filename), torch.load(term_filename)
    else:
        print("computing embedding and writing to disk")
        # note that python-based implementations are prohibitively slow for graphs of this size.
        embs, terms = embed_GO_pyG(nxGO)
        torch.save(embs, embedding_filename)
        torch.save(terms, term_filename)
    # terms already is a list
    term2emb = {
        term: embedding
        for term, embedding in zip(terms, embs)
    }
    return term2emb


if __name__ == '__main__':
    # regenerate embedding
    embed_GO(force_recompute=True)

