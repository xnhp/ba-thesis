import itertools
import os
import os.path
from enum import Enum

import networkx as nx


def is_collection_dataset(dataset) -> bool:
    path, _ = dataset
    return os.path.isdir(path)


def is_model_file(e: os.DirEntry) -> bool:
    return e.is_file() and e.name.endswith("xml")


def attrib_or_none(el, key):
    try:
        return el.attrib[key]
    except KeyError:
        return None


def groupby(input_, key_fn) -> dict:
    input_ = sorted(input_, key=key_fn)
    groups = {}
    for key, group in itertools.groupby(input_, key_fn):
        groups[key] = list(group)
    return groups

def attrview(iterable, key):
    return [i[key] for i in iterable]


def upsert_dict(target, source):
    """
    add attributes from `source` to `target` only if they do not yet exist in `target`
    :param target:
    :param source:
    :return:
    """
    d = target.copy()
    for key in source:
        if key not in target:
            d[key] = source[key]
    return d


def add_edge_safely(G, source, target, fail: bool):
    if source not in G.nodes:
        if fail:
            raise KeyError("should have found source " + source)
        else:
            print(f"edge source not found in graph: {source}")
            return
    if target not in G.nodes:
        if fail:
            raise KeyError("should have found target " + target)
        else:
            print(f"edge target not found in graph: {target}")
            return
    G.add_edge(source, target)


def init_empty_graph(model, name):
    G = nx.Graph()
    G.graph['name'] = name if name is not None else model.path
    G.graph['model'] = model
    return G


def clean_rxn_data(rxn):
    """
    obtain attributes of the reaction dict but drop references
    :param rxn:
    :return:
    """
    rxn_data = rxn.copy()
    del rxn_data['listOfReactants']
    del rxn_data['listOfProducts']
    del rxn_data['listOfModifiers']
    return rxn_data


class SpeciesClass(Enum):
    """
    Species classes expected to appear in a given model. The value of these enum fields
    corresponds to their string representation as given in the model and will be used for matching
    when parsing (can be obtained with `value`).
    ↝ cfg.dataset.possible_classes
    ↝ https://docs.python.org/3/library/enum.html
    """
    protein = 'PROTEIN'
    reaction = 'reaction'
    rna = 'RNA'
    degraded = 'DEGRADED'
    unknown = 'UNKNOWN'
    simple_molecule = 'SIMPLE_MOLECULE'
    ion = 'ION'
    gene = 'GENE'
    phenotype = 'PHENOTYPE'
    drug = 'DRUG'
    complex = 'COMPLEX'