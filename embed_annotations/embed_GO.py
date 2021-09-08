import os
from importlib.resources import files

import networkx as nx
import numpy as np
import torch

from embed_annotations.pyg_n2v import embed_GO_pyG

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

def embed_GO():
    nxGO = load_GO_graph()
    embedding_filename = os.path.join(cache_dir, "GeneOntology", "embedding.pt")
    term_filename = os.path.join(cache_dir, "GeneOntology", "term_ids.pt")
    if os.path.exists(embedding_filename):  # assume term_filename exists aswell
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


def aggregate_embeddings(df_row, term2emb, col_key):
    """
    Given a set of embeddings, aggregate them into a single one by averaging.
    :param df_row:
    :param term2emb:
    :return:
    """
    terms_of_row = df_row[col_key]
    embeddings = [term2emb[term] for term in terms_of_row]
    assert len(embeddings) > 0
    return np.mean(np.array(embeddings), axis=0)