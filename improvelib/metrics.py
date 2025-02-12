"""prediction performance metrics."""
import sys
from math import sqrt
from typing import Dict, Any

import sklearn
import numpy as np
from scipy.stats.mstats import pearsonr, spearmanr

# Import metrics from sklearn based on version
if sklearn.__version__ < "1.4.0":
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        cohen_kappa_score,
        precision_recall_curve,
        auc
    )
else:
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        root_mean_squared_error,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        cohen_kappa_score,
        precision_recall_curve,
        auc
    )


# TODO: rename str2Class to str_to_class
def str2Class(str) -> Any:
    """Convert a string to a class reference.

    Args:
        class_name (str): The name of the class to retrieve.

    Returns:
        Any: The class reference corresponding to the class name.
    """
    return getattr(sys.modules[__name__], str)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    metric_type: str, 
                    y_prob = None
                    ) -> Dict[str, float]:
    """Compute the specified set of metrics.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.
        metric_type (str): Type of metrics to compute ('classification' or 'regression').
        y_prob (np.ndaprray): Target scores made by the classification model. Optional, defaults to None.

    Returns:
        dict: A dictionary of evaluated metrics.

    Raises:
        ValueError: If an invalid metric_type is provided.
    """
    scores = {}

    if metric_type == "classification":
            metrics = ["acc", "recall", "precision", "f1", "kappa", "bacc"]
    elif metric_type == "regression":
        metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    else:
        raise ValueError(f"Invalid metric_type provided: {metric_type}. \
                         Choose 'classification' or 'regression'.")

    for mtstr in metrics:
        mapstr = mtstr
        if mapstr == "pcc":
            mapstr = "pearson"
        elif mapstr == "scc":
            mapstr = "spearman"
        elif mapstr == "r2":
            mapstr = "r_square"
        scores[mtstr] = str2Class(mapstr)(y_true, y_pred)

    if metric_type == "classification":
        if y_prob is not None:
            scores["roc_auc"] = roc_auc(y_true, y_prob)
            scores["aupr"] = aupr(y_true, y_prob)

    scores = {k: float(v) for k, v in scores.items()}
    return scores


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE).

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed MSE.
    """
    return mean_squared_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed RMSE.
    """
    if sklearn.__version__ >= "1.4.0":
        rmse = root_mean_squared_error(y_true, y_pred) # squared is deprecated
    elif sklearn.__version__ < "1.4.0" and sklearn.__version__ >= "0.22.0":
        rmse = mean_squared_error(y_true, y_pred , squared=False)
    else:
        rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient (PCC).

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed PCC.
    """
    pcc = pearsonr(y_true, y_pred)[0]
    return pcc


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman Correlation Coefficient (SCC).

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed SCC.
    """
    scc = spearmanr(y_true, y_pred)[0]
    return scc


def r_square(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 Coefficient.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed R2.
    """
    return r2_score(y_true, y_pred)


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed accuracy.
    """
    return accuracy_score(y_true, y_pred)


def bacc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed balanced accuracy.
    """
    return balanced_accuracy_score(y_true, y_pred)

def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's kappa.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed kappa.
    """
    return cohen_kappa_score(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the F1 score.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed F1 score.
    """
    return f1_score(y_true, y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed precision.
    """
    return precision_score(y_true, y_pred)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.

    Returns:
        float: The computed recall.
    """
    return recall_score(y_true, y_pred)


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Receiver Operating Characteristic AUC.

    Args:
        y_true (np.ndarray): True values to predict.
        y_prob (np.ndarray): Target scores made by the model.

    Returns:
        float: The computed ROC AUC.
    """
    return roc_auc_score(y_true, y_prob)


def aupr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall curve AUC.

    Args:
        y_true (np.ndarray): True values to predict.
        y_prob (np.ndarray): Target scores made by the model.

    Returns:
        float: The computed Precision-Recall curve AUC.
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc


