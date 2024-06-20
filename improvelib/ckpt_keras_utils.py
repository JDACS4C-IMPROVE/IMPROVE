"""
CKPT KERAS UTILS.

CANDLE checkpoint/restart utilities for Keras

"""

# Python imports
from typing import Dict

# TensorFlow imports
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Model

# CANDLE imports
from .ckpt_utils import CandleCkpt


class MultiGPUCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


class CandleCkptKeras(CandleCkpt, Callback):
    """
    Keras Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints that
    includes important metadata, particularly for restarting and tracking
    complex workflows.
    """

    def __init__(self, gParameters: Dict, logger="DEFAULT", verbose=True):
        super().__init__(gParameters, logger, verbose)

    def set_model(self, model):
        """model: The Keras model"""
        self.model = model

    # Keras Callback API:
    def on_epoch_end(self, epoch, logs=None):
        if self.save_best_metric not in logs.keys():
            raise Exception(
                (
                    "CandleCheckpointCallback: "
                    + "save_best_metric='%s' "
                    + "not in list of model metrics: %s"
                )
                % (self.save_best_metric, str(logs.keys()))
            )

        # Known metrics and direction of progress
        known_metrics = {
            "loss": "-",
            "accuracy": "+",
            "val_loss": "-",
            "val_accuracy": "+",
            "lr": "-",
        }

        if self.save_best_metric not in known_metrics.keys():
            raise Exception(
                (
                    "CandleCheckpointCallback: "
                    + "save_best_metric='%s' "
                    + "not in list of known_metrics: %s"
                )
                % (self.save_best_metric, str(known_metrics.keys()))
            )

        metric_value = logs[self.save_best_metric]
        direction = known_metrics[self.save_best_metric]
        self.ckpt_epoch(epoch, direction, metric_value)

    def write_model_backend(self, model, epoch):
        # epoch: unused by Keras
        # Keras-specific method
        model.save(self.model_file)  # save_format="h5"

    def build_model(self, model_file):
        self.model.load_weights(model_file)
