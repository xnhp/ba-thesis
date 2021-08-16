import networkx

from data.models import SBMLModel, get_dataset
from data.graphs import construct_species_graph


def print_model_summary(identifier: str):
    s = ""
    path, model_class = get_dataset(identifier)
    model: SBMLModel
    model = model_class(path)  # call specific constructor

    s += ("Model: {0} ({2}) at {1}".format(identifier, path, model_class.__name__))
    s += "\n"
    s += ("Number of species: {0}".format(len(model.species)))
    s += "\n"
    s += ("... with duplicate aliases: {0}".format(len(model.species_with_duplicate_aliases_ids)))
    s += "\n"
    s += ("Number of reactions: {0}".format(len(model.reactions)))
    s += "\n"
    rxn_without_id = len([rxn for rxn in model.reactions if rxn['id'] is None])
    s += ("... without `id` attribute: {0}".format(rxn_without_id))
    s += "\n"
    return model, s


def print_graph_summary(model: SBMLModel):
    s = ""
    # construct graph
    graph: networkx.Graph
    graph = construct_species_graph(model)

    s += ("Number of nodes: {0}".format(graph.number_of_nodes()))
    # TODO nope, they didnt
    # interesting because nielsen at al excluded species of complex type
    # and with degree < 2 (unclear whether union or intersection of these criteria)
    degrees = graph.degree(graph.nodes)  # (node, degree) tuples
    s += ("... with degree >= 2: {0}".format(
        len(
            [node for (node, degree) in degrees if degree >= SBMLModel.min_node_degree]
        )
    ))
    s += "\n"
    # complex species are already disregarded
    s += ("... with duplicate label (positive class) (after preproc):  {0}".format(len(
        [node for (node, label) in
            graph.nodes(data="node_label", default=False)
            if label == 1
         ]
    )))
    s += "\n"

    return graph, s