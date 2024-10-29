""" 
This module includes general utility functions.
"""
from typing import Dict

import numpy as np


def cast_value(s):
    """Cast a value to numeric if possible.

    Args:
        s: The value to cast.

    Returns:
        int, float, or str: The casted value if successful, otherwise the original string.
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s  # Return the original string if it's neither int nor float


def compute_performance_scores(y_true: np.array,
                               y_pred: np.array,
                               stage: str, 
                               metric_type: str, 
                               output_dir: str) -> Dict:
    """Evaluate predictions according to specified metrics.

    Args:
        y_true (np.array): Array with ground truth values.
        y_pred (np.array): Array with model predictions.
        stage (str): String specified if evaluation is with respect to val or test set.
        metric_type (str): Either classification or regression.
        output_dir (str): Directory to write results.

    Returns:
        dict: Dictionary with metrics evaluated and corresponding scores.
    """
    # Compute multiple performance scores
    scores = compute_metrics(y_true, y_pred, metric_type)

    # Add val_loss metric
    #key = f"{stage}_loss"
    #scores[key] = scores[params["loss"]]

    scores_fname = f"{stage}_scores.json"
    scorespath = Path(output_dir) / scores_fname

    with open(scorespath, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    # TODO. do we still need to print IMPROVE_RESULT?
    if stage == "val":
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
        print("Validation scores:\n\t{}".format(scores))
    elif stage == "test":
        print("Inference scores:\n\t{}".format(scores))
    else:
        print("Invalid stage: must be 'val' or 'test'.")
    return scores