import copy

import networkx
import networkx as nx

from data.models import SBMLModel, CellDesignerModel
from deprecated import deprecated

from data.util import upsert_dict, add_edge_safely, init_empty_graph, clean_rxn_data, groupby, _add_edge_root


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

def prep_res(g: nx.DiGraph) -> nx.Graph:
    # G_undirected = nx.Graph(g)
    G_undirected = g.to_undirected(as_view=True)
    G_undirected.graph['nx_multidigraph'] = g
    return G_undirected

def construct_collapsed_graph(model: SBMLModel, name=None, fail_on_unknown_edge_end=True) -> nx.Graph:

    # copy_mdl = copy.deepcopy(model)
    # copy_mdl.collapse_aliases()
    # # collapse operation only modifies aliases → might have reactions that point to aliases that are no longer there
    # G = construct_alias_graph(copy_mdl, fail_on_unknown_edge_end=False)
    # G.graph['model'] = copy_mdl
    # return G

    G = init_empty_graph(model, name)
    G.graph['name'] += " (collapsed)"
    G.graph['is_collapsed'] = True

    # dict speciesid → (complex) alias ids (top-level only)
    # this includes complex species aliases (and does not collapse them) since each CSA entry is linked to an
    # SBML species ID
    top_level_aliases_by_species = groupby(model.top_level_aliases.values(), lambda e: e['species'])

    top_level_alias_to_representative = {}
    species_to_representative = {}
    for species_id, assoc_aliases in top_level_aliases_by_species.items():
        representative = assoc_aliases[0]
        for assoc_alias in assoc_aliases:  # construct map  cleanup: move to separate loop
            top_level_alias_to_representative[assoc_alias['id']] = representative['id']
        species_to_representative[species_id] = representative
        G.add_node(representative['id'], **representative)

    # each reaction carries references to speciesAliases.
    # we need to look up what species these aliases point to and then use that to identify the proper dummy
    # TODO also consider edges that go to aliases inside a csa (edges going to not-top-level elements)
    #   should actually receive a KeyError here if so
    for rxn in model.reactions:
        rxn_data = clean_rxn_data(rxn)
        G.add_node(rxn_data['id'], **rxn_data)
        for n in rxn['listOfReactants']:
            # have exactly one alias per sbml species
            represented_species = model.top_level_aliases[n]['species']
            representative = species_to_representative[represented_species]
            add_edge_safely(G, representative['id'], rxn['id'],  fail=fail_on_unknown_edge_end)
        for n in rxn['listOfProducts']:
            represented_species = model.top_level_aliases[n]['species']
            representative = species_to_representative[represented_species]
            add_edge_safely(G, rxn['id'], representative['id'], fail=fail_on_unknown_edge_end)
        for n in rxn['listOfModifiers']:
            represented_species = model.top_level_aliases[n]['species']
            representative = species_to_representative[represented_species]
            add_edge_safely(G, representative['id'], rxn['id'],  fail=fail_on_unknown_edge_end)
            add_edge_safely(G, rxn['id'], representative['id'], fail=fail_on_unknown_edge_end)

    G.graph['top_level_alias_to_representative'] = top_level_alias_to_representative
    # G = prune_graph(G)
    return prep_res(G)


def construct_alias_graph(model: CellDesignerModel, name=None, fail_on_unknown_edge_end=True) -> nx.Graph:
    """
    Construct a graph from the given Disease Map / SBML model in which nodes are speciesAliases, linked by reactions.
    :param fail_on_unknown_edge_end: Whether an exception/warning should be thrown if an edge points to something
        that is not a node.
    :param model:
    :param name:
    :return:
    """

    G: nx.MultiDiGraph = init_empty_graph(model, name)

    for alias_id, alias_info in model.top_level_aliases.items():
        G.add_node(alias_id, **alias_info)

    # add nodes for reactions and edges from/to reactions
    for rxn in model.reactions:
        # TODO also consider edges that go to aliases inside a csa
        rxn: dict  # of info about this reaction
        rxn_data = clean_rxn_data(rxn)
        G.add_node(rxn['id'], **rxn_data)
        # we use id as key and the entire dict (containing id) as additional attributes
        # do not need to explicitly denote here that this is a reaction node because we already set its `type` attribute
        #    ↓ speciesAlias IDs
        for neighbour_id in rxn['listOfReactants']:
            _add_edge_root(G,model, neighbour_id, rxn['id'], fail_on_unknown_edge_end=fail_on_unknown_edge_end)
        for neighbour_id in rxn['listOfProducts']:
            _add_edge_root(G, model, rxn['id'], neighbour_id, fail_on_unknown_edge_end=fail_on_unknown_edge_end)
        for neighbour_id in rxn['listOfModifiers']:
            _add_edge_root(G, model, rxn['id'], neighbour_id, fail_on_unknown_edge_end=fail_on_unknown_edge_end)
            _add_edge_root(G, model, neighbour_id, rxn['id'], fail_on_unknown_edge_end=fail_on_unknown_edge_end)

    # G = prune_graph(G)
    return prep_res(G)


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
