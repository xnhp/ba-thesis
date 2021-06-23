import os.path
from importlib.resources import files


def get_dataset(identifier: str):
    """
    This is a convenience mapping so we do not have to copy/paste the syntax for loading files
    and remember the exact filename of a dataset.
    :param identifier: An identifier for the dataset
    :return: The full path to the dataset
    """
    identifier_map = {
        "AlzPathway": "alzpathway/CellDesigner SBML/12918_2012_896_MOESM1_ESM.xml",
        "PDMap": "pd_map_spring_18/PD_180412_2.xml"
    }

    filename = identifier_map[identifier]
    path = str(files("data").joinpath(filename))
    return os.path.abspath(path)


