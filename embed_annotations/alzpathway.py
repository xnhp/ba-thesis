import os
from importlib.resources import files

import pandas

from embed_annotations.embed_GO import aggregate_embeddings, embed_GO

cache_dir = files('computed')
dataset_cache_dir = os.path.join(cache_dir, "mizuno_AlzpathwayComprehensiveMap_2021", )

if __name__ == '__main__':

    # extracted from SBML with KAP workflow
    # read json file of terms from disk
    df = pandas.read_json(
        os.path.join(dataset_cache_dir, "Row1.json"),
        orient='records'
    )

    # obtain embeddings
    term2emb = embed_GO()  # obtain embeddings of GO terms

    # write results
    df['aggregated'] = df.apply(
        lambda row: aggregate_embeddings(row, term2emb, "terms"),
        axis=1
    )


    df.to_pickle(os.path.join(dataset_cache_dir, "alias-embeddings.pickle"))

    pass