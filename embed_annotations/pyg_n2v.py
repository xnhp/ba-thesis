# adapted from https://github.com/rusty1s/pytorch_geometric/
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/node2vec.py

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
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=16,
                     context_size=4, walks_per_node=1,
                     num_negative_samples=1, p=1, q=2, sparse=True).to(device)
    # increase q from default 1 for more locality

    # set num workers to 0 because pytorch multiprocessors does not work on windows
    # ↝ https://github.com/fastai/fastbook/issues/85#issuecomment-614000930
    # seems to work
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
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
    for epoch in range(1, 101):
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

