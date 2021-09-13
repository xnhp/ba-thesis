import functools
import itertools
import os
from importlib.resources import files
from typing import NewType

from deprecated.classic import deprecated
from graphgym.config import cfg
from lxml import etree
from lxml.etree import _Element
from data.util import attrib_or_none, groupby, SpeciesClass

SpeciesAliasId = NewType("SpeciesAliasId", str)
SpeciesAliasInfo = NewType("SpeciesAliasInfo", dict)
SpeciesId = NewType("SpeciesId", str)
SpeciesInfo = NewType("SpeciesInfo", dict)


class SBMLModel:
    # Static property. This is not in GraphGym configuration files because it may also be accessed from outside the GG
    # pipeline, e.g. for printing dataset summaries.
    min_node_degree = 2

    def __init__(self, filepath):
        self.path = filepath
        # cannot pickle ElementTree
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

        ## Standard SBML
        self.species_els = self.tree.findall("/model/listOfSpecies/species", self.nsmap)
        self.reaction_els = self.tree.findall("/model/listOfReactions/reaction", self.nsmap)
        # key of attribute of speciesReference that identifies the referenced species
        self.rxn_species_ref_attrib = "species"

        # to be set by implementing classes
        self.alias_groupby_attrib = None

        assert len(self.species_els) > 0
        assert len(self.reaction_els) > 0

    @functools.cached_property
    def get_top_level_alias_els(self):
        raise NotImplementedError

    def get_alias_info(self, alias: _Element) -> SpeciesAliasInfo:
        # obtain representation of alias, i.e. extract information from xml data
        # normal and complex species aliases must share the same basic attributes
        raise NotImplementedError  # to be implemented by subclasses

    @staticmethod
    def is_excluded_species(_):
        raise NotImplementedError

    def extract_speciesAlias_id(self, _):
        raise NotImplementedError

    def collapse_aliases(self):
        raise NotImplementedError
        # this is not enough for determination of duplicates, need to also re-attach edges properly, i.e.
        #   modify reactions. probably simper to construct new graph then
        # note: will modify self
        species_to_dummies = {}

        def get_dummy_alias(species: tuple[SpeciesId, SpeciesInfo]):
            species_id, species_info = species
            # in non-collapsed graphs, the id would be the speciesAlias id.
            # but we can simply use any id, as long as the species reference is proper
            some_real_alias = self.species_with_aliases[species_id][0]

            return some_real_alias['id'], some_real_alias

        self.top_level_aliases = {
                dummy_id: dummy_info
                for dummy_id, dummy_info in map(get_dummy_alias, self.species.items())
            }

    @functools.cached_property
    def top_level_aliases(self) -> dict[SpeciesAliasId, SpeciesAliasInfo]:
        return {
            alias.attrib['id']: self.get_alias_info(alias)
            for alias in self.get_top_level_alias_els
        }

    @functools.cached_property
    def species_with_aliases(self) -> dict:
        # TODO more specific type annotations for this and others
        """
        :return: dict from species id to their speciesAliases
        """
        grouped = groupby(self.top_level_aliases.values(), lambda x: x[self.alias_groupby_attrib])
        return {key: group for key, group in grouped.items()}

    @functools.cached_property
    def species_with_duplicate_aliases(self) -> dict:
        """
        :return: species that have strictly more than one alias. Each species is represented by a dict with key
        being species id and value being a list of species aliases (dicts, see self.species_aliases)
        """
        return {
            key: group for key, group in self.species_with_aliases.items() if len(group) >= 2
        }

    @functools.cached_property
    def species_with_duplicate_aliases_ids(self) -> set[str]:
        return set(self.species_with_duplicate_aliases.keys())

    @functools.cached_property
    def species_aliases_with_duplicates(self) -> set[str]:
        return set(
            # flattens the iterable of lists
            itertools.chain(*self.species_with_duplicate_aliases.values())
        )

    def get_species_info(self, species_el: _Element) -> SpeciesInfo:
        """
        Extract info of given XML element
        :param species_el:
        :return:
        """
        return SpeciesInfo({
            # species id (or should we use the `metaid` attrib instead?)
            'id': species_el.attrib['id'],
            'type': 'species',
        })

    def get_is_duplicate_as_label(self, species_id):
        return SBMLModel.is_duplicate_to_label(
            self.is_duplicate_species(species_id)
        )

    @functools.cached_property
    def species(self) -> dict[SpeciesId, SpeciesInfo]:
        """
        Return information on species as dict from species id to attributes
        :return:
        """
        return {
            species_el.attrib['id']: self.get_species_info(species_el)
            for species_el in self.species_els
        }

    @staticmethod
    def is_duplicate_to_label(value):
        """
        Convert boolean to integer labels
        """
        if value is True:
            return 1
        else:
            return 0

    def is_duplicate_species(self, species_id):
        # ↝ [[how to determine duplicates in validation networks]]
        return species_id in self.species_with_duplicate_aliases_ids

    def extraced_referenced_species_id(self, species_reference_el):
        return species_reference_el.attrib[self.rxn_species_ref_attrib]

    def extract_referenced_alias_ids(self, rxn_el, key):
        r1 = [self.extract_speciesAlias_id(species_ref_el) for species_ref_el in
                rxn_el.findall(key + "/speciesReference", self.nsmap)]
        # the below applies to listOfModifiers, we assume `speciesReference` and `modifierSpeciesReference`
        #   share a subset of core attributes and their children have the same structure.
        r2 = [self.extract_speciesAlias_id(species_ref_el) for species_ref_el in
              rxn_el.findall(key + "/modifierSpeciesReference", self.nsmap)]
        return r1 + r2

    @functools.cached_property
    def reactions(self):
        # should be standard SBML and independent of dialects?
        # TODO annotations from CellDesigner and RDF annotations ↝ read-annotations.ipynb
        return [{
                    'id': attrib_or_none(rxn_el, 'id'),
                    'class': SpeciesClass.reaction.value,
                } | {  # https://stackoverflow.com/a/26853961/156884 ;)
                    key: self.extract_referenced_alias_ids(rxn_el, key)
                    for key in ['listOfReactants', 'listOfProducts', 'listOfModifiers']
                }
                for rxn_el in self.reaction_els]


class CellDesignerModel(SBMLModel):
    def __init__(self, filepath):
        super().__init__(filepath)
        # attribute key by which to group species aliases by (determining duplicates)
        self.alias_groupby_attrib = "species"  # or layout:species


    @functools.cached_property
    def sa_root_map(self):
        """
        Map from speciesAlias id to complexSpeciesAlias id. For a given speciesAlias that is inside a complex species,
        determine its root complex (if there is no nesting of complexAliases, that is the complexAlias the speciesAlias
        lies in. Else it is lowest one).
        ↝ [[^c75162]]
        """
        def find_root_of_alias(alias_el):  # -> csa_info of root
            # find the csa that is referenced
            csa_id = alias_el.attrib['complexSpeciesAlias']
            csa_info = self.complex_aliases[csa_id]
            return find_root_of_complex(csa_info)

        def find_root_of_complex(csa_info):
            if 'complexSpeciesAlias' not in csa_info:
                return csa_info
            else:
                child = self.complex_aliases[csa_info['complexSpeciesAlias']]
                return find_root_of_complex(child)

        # speciesAliases that are contained in a complex are exactly those with an attribute "complexSpeciesAlias"
        # pointing to the parent's id
        return {
            alias_el.attrib['id']: find_root_of_alias(alias_el)
            for alias_el in self.normal_alias_els
            if 'complexSpeciesAlias' in alias_el.attrib
        }

    def is_contained(self, alias_id):
        return alias_id in self.sa_root_map

    @functools.cached_property
    def normal_alias_els(self):
        return self.tree.findall(
            "/model/annotation/celldesigner:extension/celldesigner:listOfSpeciesAliases/celldesigner:speciesAlias",
            self.nsmap)

    @functools.cached_property
    def complex_alias_els(self):
        return self.tree.findall(
            "/model/annotation/celldesigner:extension/celldesigner:listOfComplexSpeciesAliases/celldesigner:complexSpeciesAlias",
            self.nsmap)


    @functools.cached_property
    def complex_aliases(self):
        return {
            csa.attrib['id']: {**csa.attrib}
            for csa in self.complex_alias_els
        }

    @functools.cached_property
    def get_top_level_alias_els(self):
        top_level_alias_els = [e for e in self.normal_alias_els if not 'complexSpeciesAlias' in e.attrib]
        top_level_complex_alias_els = [e for e in self.complex_alias_els if not 'complexSpeciesAlias' in e.attrib]
        return top_level_alias_els + top_level_complex_alias_els

    def get_species_info(self, species_el) -> SpeciesInfo:
        d = super().get_species_info(species_el)
        d['class'] = species_el.find(
            "annotation/celldesigner:extension/celldesigner:speciesIdentity/celldesigner:class",
            self.nsmap).text
        # TODO annotations, ↝ read-annotations.ipynb ↝ [[exploit annotations for features]]
        return d

    def extract_GO_annotations(self, species_el):
        version_of_els = species_el.find(
            "annotation/celldesigner:extension/rdf:RDF/rdf:Description/bqbiol:isVersionOf",
            self.nsmap
        )
        for version_of_el in version_of_els:
            li = version_of_el.find("rdf:Bag/rdf:li", self.nsmap)
            go_term = li.attrib['rdf:resource']

    def get_alias_info(self, alias: _Element):
        # d = dict(alias.attrib)
        d = {
            # this set of attributes is shared by both normal and complex species aliases
            'id': alias.attrib['id'],
            'species': alias.attrib['species']
        }
        # additionally add information on the species this alias represents
        species_info = self.species[alias.attrib['species']]
        for key in species_info:  # do not overwrite attributes with coinciding names
            if key not in d:
                d[key] = species_info[key]
        return d

    @staticmethod
    def is_excluded_species(d):
        return cfg.dataset.exclude_complex_species and d['class'] == "COMPLEX"

    def extract_speciesAlias_id(self, species_reference: _Element):
        """
        find the corresponding speciesAlias id given a speciesReferenceElement. In CellDesigner models,
        the structure is
        reaction
            listOfReactants / listOfProducts
                speciesReference
                    annotation
                        celldesigner:extension
                            celldesigner:alias
                                (contained text is speciesAlias id)
        Note that in case of listOfModifiers, `modifierSpeciesReference` is used instead of `speciesReference.
        The structure of the contained elements seems to be the same.
        :param species_reference:
        :return:
        """
        return species_reference.find("annotation/celldesigner:extension/celldesigner:alias", self.nsmap).text


@deprecated(reason="needs work after other adjustments and not sure right now if we will ever need this again")
class SBMLLayoutModel(SBMLModel):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.nsmap['layout'] = "http://www.sbml.org/sbml/level3/version1/layout/version1"
        # for whatever reason the layout namespace is prefixed to attribute keys
        self.attrib_ns_prefix = "{http://www.sbml.org/sbml/level3/version1/layout/version1}"
        self.alias_groupby_attrib = self.attrib_ns_prefix + "species"

    # ↝ c33436
    def get_species_info(self, species_el):
        d = super().get_species_info(species_el)
        # this kind of data format (at least the ReconMap example) does not seem to have class annotations
        d['class'] = "unknown"  # TODO
        return d


def get_dataset(identifier: str) -> tuple[str, SBMLModel]:
    """
    This is a convenience mapping so we do not have to copy/paste the syntax for loading files
    and remember the exact filename of a dataset.
    ↝ ORIGIN.txt in data subdirectories
    :param identifier: An identifier for the dataset
    :return: The full path to the dataset
    """
    identifier_map = {
        "AlzPathwayReorg": (
            # note that this describes a collection of networks
            "AlzPah_reorganisation_steps",  # is a directory
            CellDesignerModel
        ),
        "AlzPathway": (
            # from [[mizuno_AlzPathwayComprehensiveMap_2021]]
            "alzpathway/CellDesigner SBML/12918_2012_896_MOESM1_ESM.xml",
            CellDesignerModel
        ),
        "PDMap": (
            "pd_map_spring_18/PD_180412_2.xml",
            CellDesignerModel
        ),
        "PDMap19": (
            # have GO/BP annotations for this one
            "pd_map_autmn_19/PD_190925_1.xml",
            CellDesignerModel
        ),
        "NF-kB": (
            "examples/PD_Fig 8 NF-kappaB updated.xml",
            CellDesignerModel
        ),
        "multiple-split": (
            "examples/multiple-split.xml",
            CellDesignerModel
        ),
        "invalid-split": (
            "examples/invalid-split.xml",
            CellDesignerModel
        ),
        # ↝ c33436
        # "ReconMap": (
        #     "ReconMap/ReconMap-2.01-SBML3-Layout-Render/ReconMap-2.01-SBML3-Layout-Render.xml",
        #     SBMLLayoutModel
        # ),
        "ReconMapOlder": (
            "ReconMap/ReconMap-2.01/ReconMap-2.01.xml",
            CellDesignerModel
        )
    }

    try:
        match = identifier_map[identifier]
    except KeyError:
        match = (
            "examples/" + identifier + ".xml",
            CellDesignerModel
        )
    path = str(files("data").joinpath(match[0]))
    return os.path.abspath(path), match[1]
