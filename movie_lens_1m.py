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
        hidden_dims = [128, 128],
        epochs=500,
        batch_size=16,

        # regularization
        # kernel_regularizer=None,
        kernel_regularizer=regularizers.l2(0.00005),
        # kernel_regularizer=regularizers.l1(0.00001), # 0.9771

        # dropout
        drop_rate=0.3,

        # activation
        # activation='sigmoid', batch_norm=True,
        activation='relu', batch_norm=True,

        # drop input
        input_drop_rate = 0.3,
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
        callbacks=[callbacks.TensorBoard('./output/movie_lens_1m/final')]
    )
    model.save('./output/movie_lens_1m/models/', overwrite=True)

    # inference
    # model.load_weights('./output/movie_lens_1m/models/model.h5')    
#   sample = test_set[:1]
#   prediction = model.predict(sample)
#   print(sample)
#   print(prediction)
