import numpy as np


def eval_gini(y_true, y_pred):
    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0]
    l_mid = np.linspace(1 / n_samples, 1, n_samples)

    pred_order = y_true[y_pred.argsort()]
    l_pred = np.cumsum(pred_order) / np.sum(pred_order)
    g_pred = np.sum(l_mid - l_pred)

    true_order = y_true[y_true.argsort()]
    l_true = np.cumsum(true_order) / np.sum(true_order)
    g_true = np.sum(l_mid - l_true)

    return g_pred / g_true
