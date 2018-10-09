from .base import BaseModel 
from keras.layers import (
    Input, Dense
)
from keras.models import Model
from keras import initializers, regularizers


class AutoEncoder(BaseModel):
    """AutoEncoder for Collaborative Filtering

    The model tries to reconstruct the ratings given by a user or
    with respect to an item.
    """ 
    hparam_check_list = [
        'input_dim', 'hidden_dims', 'activation',
        'epochs', 'batch_size', 'kernel_initializer',
        'kernel_regularizer',
    ]
    hparam_default_dict = {
        'activation': 'sigmoid',
        'epochs': 10,
        'batch_size': 32,
        'kernel_initializer': initializers.random_normal(stddev=0.05),
        'kernel_regularizer': None,
    }

    def _build(self):
        hp = self.hparam

        # AutoEncoder 
        input_tensor = x = Input(shape=(hp.input_dim,))
        for nb_hidden in hp.hidden_dims:
            layer = Dense(
                nb_hidden,
                activation=hp.activation,
                kernel_initializer=hp.kernel_initializer,
                kernel_regularizer=hp.kernel_regularizer
            )
            x = layer(x)
        output_tensor = Dense(
                hp.input_dim,
                kernel_initializer=hp.kernel_initializer,
                kernel_regularizer=hp.kernel_regularizer
        )(x)

        # assigning attributes
        self.model = Model(input_tensor, output_tensor)

    def fit(self, train_set, valid_set=None, **kwargs):
        hp = self.hparam

        self.model.fit(
            x=train_set, y=train_set,
            epochs=hp.epochs,
            validation_data=(valid_set, valid_set),
            **kwargs
        )

    def predict(self, samples): 
        return self.model.predict(samples)

    # def encode(self, samples):
    #     pass


import keras.backend as K
def nonzero_mse(y_true, y_pred):
    """MSE that ignores zero values, assuming that there is at least
    one nonzero value per row
    """
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    count = K.sum(mask, 1)
    se = K.sum(K.square(y_true-y_pred)*mask, 1)
    mse = se / count
    return K.mean(mse)

def nonzero_rmse(y_true, y_pred):
    """RMSE that ignores zero values, assuming that there is at least
    one nonzero value per row
    """
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    count = K.sum(mask, 1)
    se = K.sum(K.square(y_true-y_pred)*mask, 1)
    rmse = K.sqrt(se / count)
    return K.mean(rmse)

def nonzero_batch_mse(y_true, y_pred):
    """MSE that ignores zero values and averages over a batch, assuming
    that there is at least one nonzero value per batch.
    """
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    count = K.sum(mask)
    se = K.sum(K.square(y_true-y_pred)*mask)
    mse = se / count
    return mse

def nonzero_batch_rmse(y_true, y_pred):
    """RMSE that ignores zero values and averages over a batch, assuming
    that there is at least one nonzero value per batch.
    """
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    count = K.sum(mask)
    se = K.sum(K.square(y_true-y_pred)*mask)
    rmse = K.sqrt(se / count)
    return rmse
