import numpy as np


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