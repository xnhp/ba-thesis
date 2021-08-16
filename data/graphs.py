import copy

import networkx
import networkx as nx

from data.models import SBMLModel
from deprecated import deprecated

from data.util import upsert_dict, add_edge_safely, init_empty_graph, clean_rxn_data


@deprecated(reason="This interpretation is not considered ↝ [[^181948]]")
def construct_species_graph(model: SBMLModel, name=None) -> nx.Graph:
    """
    Construct a graph from the given Disease Map / SBML Model in which nodes are species, linked by reactions.
    :param model:
    :param name:
    :return:
    """
    raise NotImplementedError("deprecated")
    G = init_empty_graph(model, name)

    # fetch full info about species and already add as nodes
    for species_id, species_info in model.species.items():
        G.add_node(species_id, species_info)

    # add nodes for reactions and edges from/to reactions
    for rxn in model.reactions:
        rxn: dict  # of info about this reaction
        rxn_data = rxn.copy()
        del rxn_data['reactants']
        del rxn_data['products']
        del rxn_data['modifiers']
        reaction_node = G.add_node(rxn['id'], **rxn_data)
        # we use id as key and the entire dict (containg id) as additional attributes
        # do not need to explicitly denote here that this is a reaction node because we already set its `type` attribute
        for neighbour_id in rxn['reactants'] + rxn['products'] + rxn['modifiers']:
            if neighbour_id in G.nodes:  # `add_edge` adds nodes if they don't exist yet
                # there might be excluded species that do not appear in model.species
                # but are referenced in a reaction in model.reactions
                G.add_edge(rxn['id'], neighbour_id)  # disregard direction for now

    G = prune_graph(G)

    return G


def construct_collapsed_graph(model: SBMLModel, name=None, fail_on_unknown_edge_end=True) -> nx.Graph:

    # copy_mdl = copy.deepcopy(model)
    # copy_mdl.collapse_aliases()
    # # collapse operation only modifies aliases → might have reactions that point to aliases that are no longer there
    # G = construct_alias_graph(copy_mdl, fail_on_unknown_edge_end=False)
    # G.graph['model'] = copy_mdl
    # return G

    G = init_empty_graph(model, name)

    species_to_representative = {}
    # for each species, introduce exactly one speciesAlias (s.t. we have the same data structure)
    for species_id, species_info in model.species.items():
        some_real_alias = model.species_with_aliases[species_id][0]
        species_to_representative[species_id] = some_real_alias
        # what information/attributes to set on a dummy alias? let's set custom id and beyond that use the attribs
        #   of any alias it is a dummy for
        G.add_node(some_real_alias['id'], **some_real_alias)

    # each reaction carries references to speciesAliases.
    # we need to look up what species these aliases point to and then use that to identify the proper dummy
    for rxn in model.reactions:
        rxn_data = clean_rxn_data(rxn)
        G.add_node(rxn_data['id'], **rxn_data)
        for neighbour_id in rxn['listOfReactants'] + rxn['listOfProducts'] + rxn['listOfModifiers']:
            represented_species = model.aliases[neighbour_id]['species']
            representative = species_to_representative[represented_species]
            add_edge_safely(G, rxn['id'], representative['id'], fail=fail_on_unknown_edge_end)

    G = prune_graph(G)

    return G


def construct_alias_graph(model: SBMLModel, name=None, fail_on_unknown_edge_end=True) -> nx.Graph:
    """
    Construct a graph from the given Disease Map / SBML model in which nodes are speciesAliases, linked by reactions.
    :param fail_on_unknown_edge_end: Whether an exception/warning should be thrown if an edge points to something
        that is not a node.
    :param model:
    :param name:
    :return:
    """
    G = init_empty_graph(model, name)

    for alias_id, alias_info in model.aliases.items():
        G.add_node(alias_id, **alias_info)

    # add nodes for reactions and edges from/to reactions
    for rxn in model.reactions:
        rxn: dict  # of info about this reaction
        rxn_data = clean_rxn_data(rxn)
        G.add_node(rxn['id'], **rxn_data)
        # we use id as key and the entire dict (containing id) as additional attributes
        # do not need to explicitly denote here that this is a reaction node because we already set its `type` attribute
        #    ↓ speciesAlias IDs
        for neighbour_id in rxn['listOfReactants'] + rxn['listOfProducts'] + rxn['listOfModifiers']:
            add_edge_safely(G, rxn['id'], neighbour_id, fail=fail_on_unknown_edge_end)

    G = prune_graph(G)

    return G


def prune_graph(G: networkx.Graph):
    # low_deg_nodes = [node for (node, degree) in G.degree if degree < SBMLModel.min_node_degree]
    # G.remove_nodes_from(low_deg_nodes)
    # note that this does not necessarily mean that after this G will contain no nodes of low degree
    # we have to accept this...
    # ... but we need to remove isolated nodes because we cannot compute neighbourhood statistics for them
    #     and in any case duplicating isolated nodes seems out of scope
    # no_deg_nodes = [node for (node, degree) in G.degree if degree == 0]
    # G.remove_nodes_from(no_deg_nodes)
    # degrees = [deg for (node, deg) in G.degree()]
    ## assert max(degrees) >= SBMLModel.min_node_degree
    # assert min(degrees) > 0
    return G
