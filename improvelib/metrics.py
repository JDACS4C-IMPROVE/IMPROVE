"""Functionality for Computing Metrics in IMPROVE."""

import sys

from scipy.stats.mstats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error


def str2Class(str):
    return getattr(sys.modules[__name__], str)


def compute_metrics(y_true, y_pred, metrics):
    """Compute the specified set of metrics.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.
    metrics: python list
        List of metrics to compute.

    Returns
    -------
    eval: python dictionary
        A dictionary of evaluated metrics.
    """
    scores = {}
    for mtstr in metrics:
        mapstr = mtstr
        if mapstr == "pcc":
            mapstr = "pearson"
        elif mapstr == "scc":
            mapstr = "spearman"
        elif mapstr == "r2":
            mapstr = "r_square"
        scores[mtstr] = str2Class(mapstr)(y_true, y_pred)

    scores = {k: float(v) for k, v in scores.items()}

    return scores


def mse(y_true, y_pred):
    """Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to MSE. If several outputs, errors of all outputs are averaged with uniform weight.
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse


def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to RMSE. If several outputs, errors of all outputs are averaged with uniform weight.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse


def pearson(y_true, y_pred):
    """Compute Pearson Correlation Coefficient (PCC).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to PCC.
    """
    pcc = pearsonr(y_true, y_pred)[0]
    return pcc


def spearman(y_true, y_pred):
    """Compute Spearman Correlation Coefficient (SCC).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to SCC.
    """
    scc = spearmanr(y_true, y_pred)[0]
    return scc


def r_square(y_true, y_pred):
    """Compute R2 Coefficient.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to R2. If several outputs, scores of all outputs are averaged with uniform weight.
    """

    return r2_score(y_true, y_pred)
