import os
import sqlite3
from importlib.resources import files

import mygene
import numpy as np
import pandas as pd  # from china
from matplotlib import pyplot as plt

from embed_annotations.embed_GO import aggregate_embeddings
from embed_annotations.pyg_n2v import embed_GO

cache_dir = files('computed')
db_path = os.path.join(cache_dir, "GeneOntology", 'mygene-cache.db')
db_con = sqlite3.connect(db_path)
mg = mygene.MyGeneInfo()
mg.set_caching(db_path)


def read_minerva_export():
    # read export of annotations from minerva
    # tsv file with columns type, element external id (species alias), ensebml, entrez, hgnc
    # typo intentional
    # TODO move this to computed?
    path = os.path.join(files("data"), 'pd_map_autmn_19', 'pd_map_autumn_19-elementExport.txt')
    # with open(path) as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     next(reader)  # skip first row (contains headers)
    #     for row in reader:
    #         print(row)
    df = pd.read_csv(path, sep='\t', header=0)
    # export has trailing `,` on IDs
    df = df.dropna(subset=['Ensembl'])
    df['Ensembl'] = df['Ensembl'].str[:-1]
    # TODO dont need these below
    df['Entrez Gene'] = df['Entrez Gene'].str[:-1]
    df['HGNC'] = df['HGNC'].str[:-1]
    return df


def query_GO_terms(df):
    pd.options.mode.chained_assignment = None  # default='warn'
    response = mg.getgenes(df['Entrez Gene'], fields="go")
    df['go.BP'] = response

    not_mapped = []

    def process_indiv_res(row):
        response = row['go.BP']
        if 'go' not in response or 'notfound' in response:
            not_mapped.append(response)
            return np.NaN
        else:
            # extract BP part
            if 'BP' not in response['go']:  # can have GO annots but none for BP subgraph
                not_mapped.append(response)
                return np.NaN
            bp_content = response['go']['BP']  # can be list if multiple annotations or dict if only one
            bp_terms = bp_content if type(bp_content) is not dict else [bp_content]
            bp_terms = list(filter(is_valid_term, bp_terms))
            bp_terms = [term['id'] for term in bp_terms]
            return bp_terms if is_valid_term_set(bp_terms) else np.NaN

    df['go.BP'] = df.apply(lambda row: process_indiv_res(row), axis=1)
    df = df.dropna(subset=['go.BP'])

    return df

def is_valid_term_set(terms: list):
    # return len(terms) > 0
    # could also disregard species with high number of associations?
    return len(terms) < 40 and len(terms) > 0
    # from visual inspection of histogram to elim. "outliers", this is very much in connection with the
    # implementation of is_valid term which determines how many terms we find for a species
    # motivation for this cutoff is to keep the number of associated terms for a species small

def is_valid_term(term: dict):
    # ↝ http://geneontology.org/docs/guide-go-evidence-codes/
    # ↝ [[ruiz_identification_2021]]
    # technically dont need to check for this since we only consider BO subgraph anyway
    return term['gocategory'] == 'BP' \
        and term['evidence'] in ["EXP", "IDA", "IMP", "IGI", "HTP", "HDA", "HMP", "HGI"]  # based on ruiz et al
    # and term['qualifier'] not in ['involved_in'] \   # only ~1k left, cannot do this
    # and term['evidence'] not in ["ND"]

# too slow
# def embed_GO_graph(bp_comp, embedding_filename):
#     print("computing embedding...")
#     n2v = Node2Vec(bp_comp,
#                    dimensions=64,
#                    walk_length=30,
#                    num_walks=8,  # number of walks *per node*
#                    workers=1)
#
#     mdl = n2v.fit(window=10, min_count=2, batch_words=4)
#     mdl.wv.save_word2vec_format(embedding_filename)
#     # mdl.wv.save(embedding_filename + ".wordvectors")  # can also do this
#     return mdl.wv
#

def analyze_queried_terms(df):
    # histogram of number of found GO/BP terms for each species.
    lens = df.apply(lambda row: len(row['go.BP']), axis=1)
    plt.hist(lens, bins=int(len(lens) / 10))
    assert not any(lens == 0)  # assume we have already filtered out those for which we have not found (valid) GO terms
    plt.title(f"number of species: {len(lens)}")
    plt.show()

if __name__ == '__main__':
    # Obtain and aggregate GO term embeddings per species alias for
    # pd_map_autumn_19 (annotations based on minerva export)

    df = read_minerva_export()

    df = query_GO_terms(df)
    # analyze_queried_terms(df)

    # GO term → its embedding
    term2emb = embed_GO()  # obtain embeddings of GO terms

    df['aggregated'] = df.apply(
        lambda row: aggregate_embeddings(row, term2emb, 'go.BP'),
        axis=1
    )

    df.to_pickle(os.path.join(cache_dir, "pd_map_autumn_19", "alias-embeddings.pickle"))
    # df that contains column 'aggregated' that holds list/ndarray/tensor of aggregated word2vec embedding

    # TODO t-SNE embedding of GO embeddings, or subset of species to consider
    #   would also be interesting to assess how much the two embeddings overlap in GO terms

    print("done")
