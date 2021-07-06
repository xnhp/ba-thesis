import functools
import itertools
import os.path
from importlib.resources import files

import networkx
from graphgym.config import cfg
from lxml import etree


class SBMLModel:
    def __init__(self, filepath):
        self.path = filepath
        self.tree = etree.parse(filepath)
        self.root = self.tree.getroot()
        # need to explicitly add these namespaces
        self.nsmap = self.root.nsmap.copy()
        self.nsmap['rdf'] = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        self.nsmap['dc'] = "http://purl.org/dc/elements/1.1/"
        self.nsmap['dcterms'] = "http://purl.org/dc/terms/"
        self.nsmap['vCard'] = "http://www.w3.org/2001/vcard-rdf/3.0#"
        self.nsmap['bqbiol'] = "http://biomodels.net/biology-qualifiers/"
        self.nsmap['bqmodel'] = "http://biomodels.net/model-qualifiers/"

        ## Standard SBML, applies to all dialects
        self.species_els = self.tree.findall("/model/listOfSpecies/species", self.nsmap)
        self.reaction_els = self.tree.findall("/model/listOfReactions/reaction", self.nsmap)
        # key of attribute of speciesReference that identifies the referenced species
        self.rxn_species_ref_attrib = "species"

        # to be set by implementing classes
        self.alias_groupby_attrib = None
        self.species_alias_els = None

        assert len(self.species_els) > 0
        assert len(self.reaction_els) > 0

    @functools.cached_property
    def species_aliases(self) -> list[dict]:
        """
        :return: List of dicts representing species aliases
        """
        # Note that this does not consider `listOfComplexSpeciesAliases`. Omitted for now because we are currently not
        # considering complex species in the first place.
        return [alias.attrib for alias in self.species_alias_els]

    @functools.cached_property
    def duplicate_aliases(self) -> dict:
        """
        :return: species that have strictly more than one alias. Each species is represented by a dict with key
        being species id and value being a list of species aliases (dicts, see self.species_aliases)
        """
        # sort by key required before groupby
        aliases_sorted = sorted(self.species_aliases, key=lambda x: x[self.alias_groupby_attrib])
        # could also operate on the iterator that groupby returns but
        # for values to persist (not be shared), we have to put them into a list
        grouped = {}
        for key, group in itertools.groupby(aliases_sorted, lambda x: x[self.alias_groupby_attrib]):
            grouped[key] = list(group)
        duplicates = {key: group for key, group in grouped.items() if len(group) > 1}
        return duplicates

    @functools.cached_property
    def duplicate_aliases_ids(self) -> set[str]:
        return set(self.duplicate_aliases.keys())

    def get_species_dict(self, species):
        d = {'id': species.attrib['id'], 'type': 'species', 'is_duplicate': self.is_duplicate_species(species)}
        # species id (or should we use the `metaid` attrib instead?)
        d['node_label'] = d['is_duplicate']  # GG expects this name
        return d

    @functools.cached_property
    def species(self) -> list[dict]:

        return [
            d for d in [
                self.get_species_dict(species) for species in self.species_els
            ]
            if not self.is_excluded_species(d)
        ]

    def is_duplicate_species(self, species):
        # ↝ [[how to determine duplicates in validation networks]]
        return species.attrib['id'] in self.duplicate_aliases_ids

    @staticmethod
    def is_excluded_species(d):
        return cfg.dataset.exclude_complex_species and d['class'] == "COMPLEX"

    @functools.cached_property
    def reactions(self):
        def extract_species_reference(el):
            return el.attrib[self.rxn_species_ref_attrib]

        r = []
        for rxn in self.reaction_els:
            d = {'id': rxn.attrib['id'], 'class': 'reaction', 'node_label': 0,
                 'reactants': [extract_species_reference(el) for el in
                               rxn.findall("listOfReactants/speciesReference", self.nsmap)],
                 'products': [extract_species_reference(el) for el in
                              rxn.findall("listOfProducts/speciesReference", self.nsmap)],
                 'modifiers': [extract_species_reference(el) for el in
                               rxn.findall("listOfModifiers/speciesReference", self.nsmap)]}
            # should be standard SBML and independent of dialects?
            # TODO need to set node_label here aswell?
            # TODO annotations from CellDesigner and RDF annotations ↝ read-annotations.ipynb
            r.append(d)
        return r


class CellDesignerModel(SBMLModel):
    def __init__(self, filepath):
        super().__init__(filepath)
        # attribute key by which to group species aliases by (determining duplicates)
        self.species_alias_els = self.tree.findall(
            "/model/annotation/celldesigner:extension/celldesigner:listOfSpeciesAliases/celldesigner:speciesAlias",
            self.nsmap)
        self.alias_groupby_attrib = "species"  # or layout:species

    def get_species_dict(self, species):
        d = super().get_species_dict(species)
        # species/node class (as per [[^2e2cfd]])
        d['class'] = species.find("annotation/celldesigner:extension/celldesigner:speciesIdentity/celldesigner:class",
                                  self.nsmap).text
        # TODO annotations, ↝ read-annotations.ipynb ↝ [[exploit annotations for features]]
        return d


class SBMLLayoutModel(SBMLModel):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.nsmap['layout'] = "http://www.sbml.org/sbml/level3/version1/layout/version1"
        self.species_alias_els = self.tree.findall(
            "/model/layout:listOfLayouts/layout:layout/layout:listOfSpeciesGlyphs/layout:speciesGlyph", self.nsmap)
        # for whatever reason the layout namespace is prefixed to attribute keys
        self.attrib_ns_prefix = "{http://www.sbml.org/sbml/level3/version1/layout/version1}"
        self.alias_groupby_attrib = self.attrib_ns_prefix + "species"


def get_dataset(identifier: str) -> tuple[str, SBMLModel]:
    """
    This is a convenience mapping so we do not have to copy/paste the syntax for loading files
    and remember the exact filename of a dataset.
    :param identifier: An identifier for the dataset
    :return: The full path to the dataset
    """
    # need to do local import
    identifier_map = {
        "AlzPathway": (
            "alzpathway/CellDesigner SBML/12918_2012_896_MOESM1_ESM.xml",
            CellDesignerModel
        ),
        "PDMap": (
            "pd_map_spring_18/PD_180412_2.xml",
            CellDesignerModel
        ),
        "ReconMap": (
            "ReconMap/ReconMap-2.01-SBML3-Layout-Render/ReconMap-2.01-SBML3-Layout-Render.xml",
            SBMLLayoutModel
        ),
        "ReconMapOlder": (
            "ReconMap/ReconMap-2.01/ReconMap-2.01.xml",
            CellDesignerModel
        )
    }

    match = identifier_map[identifier]
    path = str(files("data").joinpath(match[0]))
    return os.path.abspath(path), match[1]


def load_dataset(identifier: str) -> tuple[SBMLModel, networkx.Graph]:
    path, model_class = get_dataset(identifier)
    model = model_class(path)
    from graphgym.contrib.loader.SBML import graph_from_model
    return model, graph_from_model(model)


def print_model_summary(identifier: str):
    model, graph = load_dataset(identifier)
    print(
        "Number of species: {0}".format(len(model.species))
    )
    print(
        "Number of duplicate species: {0}".format(len(model.duplicate_aliases_ids))
    )