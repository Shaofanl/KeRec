from kerec.models.ae import (
    nonzero_mse, nonzero_rmse,
    nonzero_batch_mse, nonzero_batch_rmse
)
import keras.backend as K

import numpy as np


def test_nonzero_mse():
    y_true = np.random.randint(0, 5, (7, 10))
    y_pred = np.random.randint(0, 5, (7, 10))

    loss = K.eval(nonzero_mse(K.variable(y_true), K.variable(y_pred)))
    mask = y_true != 0
    est = (((y_true-y_pred)**2*mask).sum(1) / (mask).sum(1)).mean()
    assert np.isclose(loss, est)

def test_nonzero_rmse():
    y_true = np.random.randint(0, 5, (7, 10))
    y_pred = np.random.randint(0, 5, (7, 10))

    loss = K.eval(nonzero_rmse(K.variable(y_true), K.variable(y_pred)))
    mask = y_true != 0
    est = ((((y_true-y_pred)**2*mask).sum(1) / (mask).sum(1))**.5).mean()
    assert np.isclose(loss, est)

def test_nonzero_batch_mse():
    y_true = np.random.randint(0, 5, (7, 10))
    y_pred = np.random.randint(0, 5, (7, 10))

    loss = K.eval(nonzero_mse(K.variable(y_true), K.variable(y_pred)))
    mask = y_true != 0
    est = ((y_true-y_pred)**2*mask).sum() / (mask).sum()
    assert np.isclose(loss, est, rtol=1e-01, atol=1e-02)

def test_nonzero_batch_rmse():
    y_true = np.random.randint(0, 5, (7, 10))
    y_pred = np.random.randint(0, 5, (7, 10))

    loss = K.eval(nonzero_rmse(K.variable(y_true), K.variable(y_pred)))
    mask = y_true != 0
    est = (((y_true-y_pred)**2*mask).sum() / (mask).sum())**.5
    assert np.isclose(loss, est, rtol=1e-01, atol=1e-02)
