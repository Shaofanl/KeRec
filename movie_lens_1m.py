from kerec.models.ae import (
    AutoEncoder, nonzero_mse, nonzero_rmse,
    nonzero_batch_mse, nonzero_batch_rmse
)
from kerec.utils import HyperParameterSet
from kerec.utils.data import sparse_rating_to_dense

from keras.optimizers import Adam
from keras import initializers, regularizers, callbacks

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # data loading
    filename = './dataset/ml-1m/ratings.dat'
    records = (
        pd.read_csv(filename, sep='::', header=None)
            .values[:, :-1]
            .astype(np.uint32)
    )
    ratings = sparse_rating_to_dense(records, 'int')
    print('Ratings shape: {}'.format(ratings.shape))
    train_set, test_set = train_test_split(ratings, test_size=0.33, random_state=32)

    # setup hyperparameters
    hparam = HyperParameterSet(
        input_dim = 3953,
        hidden_dims = [128, 64, 128],
        epochs=2000,
        batch_size=16,
        # kernel_regularizer=None,
        kernel_regularizer=regularizers.l2(0.00005),
        # kernel_regularizer=regularizers.l1(0.00001), # 0.9771
    )

    # build model
    model = AutoEncoder(hparam)
    model.compile(
        optimizer=Adam(lr=0.001),
        loss=nonzero_batch_mse,
        metrics=[nonzero_batch_rmse]
    )

    # train model
    model.fit(
        train_set=train_set,
        valid_set=test_set,
        callbacks=[callbacks.TensorBoard('./output/movie_lens_1m/layer/128_64_128')]
    )
