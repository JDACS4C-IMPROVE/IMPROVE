"""Functionality for Computing Metrics in IMPROVE."""

import sys
import sklearn
from math import sqrt

from scipy.stats.mstats import pearsonr, spearmanr

if sklearn.__version__ < "1.4.0":
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
else:
    from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score


def str2Class(str):
    return getattr(sys.modules[__name__], str)


def compute_metrics(y_true, y_pred, metric_type):
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
    if metric_type == "classification":
        metrics = ["acc", "recall", "precision", "f1", "auc", "aupr"]
    elif metric_type == "regression":
        metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    else:
        print("Invalid metric_type")

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
    #rmse = root_mean_squared_error(y_true, y_pred)
    if sklearn.__version__ >= "1.4.0":
        rmse = root_mean_squared_error(y_true, y_pred) # squared is deprecated
    elif sklearn.__version__ < "1.4.0" and sklearn.__version__ >= "0.22.0":
        rmse = mean_squared_error(y_true, y_pred , squared=False)
    else:
        rmse = sqrt(mean_squared_error(y_true, y_pred))
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


def acc(y_true, y_pred):
    """Compute accuracy.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to accuracy.
    """

    return accuracy_score(y_true, y_pred)


def bacc(y_true, y_pred):
    """Compute balanced accuracy.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to balanced accuracy.
    """

    return balanced_accuracy_score(y_true, y_pred)


def f1(y_true, y_pred):
    """Compute the F1 score.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to the F1 score.
    """

    return f1_score(y_true, y_pred)


def precision(y_true, y_pred):
    """Compute precision.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to precision.
    """

    return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    """Compute recall.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to recall.
    """

    return recall_score(y_true, y_pred)


def auc(y_true, y_pred):
    """Compute Receiver Operating Characteristic AUC.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to ROC AUC.
    """

    return roc_auc_score(y_true, y_pred)


def aupr(y_true, y_pred):
    """Compute Precision-Recall curve AUC.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to Precision-Recall curve AUC.
    """

    return average_precision_score(y_true, y_pred)
