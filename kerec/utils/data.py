import numpy as np


def sparse_rating_to_dense(records, dtype=np.float, remove_empty=True):
    """Transform a set of sparse (user_id, item_id, rating) records to a dense rating matrix 

    Input:
        records: record in the format of (user_id, item_id, rating)

    Output:
        ratings: a dense matrix and cell (user_id, item_id) contains the rating
    """

    max_user_id = records[:, 0].max()
    max_item_id = records[:, 1].max()

    ratings = []

    for user_id in range(max_user_id+1):
        item_ids = records[records[:, 0]==user_id, 1]
        rate_val = records[records[:, 0]==user_id, 2]

        row = np.zeros(max_item_id+1, dtype=dtype)
        row[item_ids] = rate_val
        if remove_empty and row.sum() == 0:
            continue
        ratings.append(row)
    return np.array(ratings)     
