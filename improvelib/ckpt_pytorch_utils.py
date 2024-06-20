"""
CKPT PYTORCH UTILS.

CANDLE checkpoint/restart utilities for PyTorch
"""

# Python imports
from typing import Dict

# PyTorch imports
import torch

# CANDLE imports
from .ckpt_utils import CandleCkpt


class CandleCkptPyTorch(CandleCkpt):
    """
    PyTorch Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints that
    includes important metadata, particularly for restarting and tracking
    complex workflows.
    """

    def __init__(self, gParams: Dict, logger="DEFAULT", verbose=True):
        super().__init__(gParams, logger, verbose)

    def set_model(self, model):
        """
        model: A dict with the model {'model':model, 'optimizer':optimizer}
        """
        self.model = model

    def ckpt_epoch(self, epoch, metric_value):
        """The PyTorch training loop should call this each epoch"""
        direction = "-"
        super().ckpt_epoch(epoch, direction, metric_value)

    def write_model_backend(self, model, epoch):
        m = self.model["model"]
        o = self.model["optimizer"]

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": m.state_dict(),
                "optimizer_state_dict": o.state_dict(),
                "loss": 0,
            },
            self.model_file,
        )

    def build_model(self, model_file):
        m = self.model["model"]
        o = self.model["optimizer"]

        checkpoint = torch.load(model_file)
        m.load_state_dict(checkpoint["model_state_dict"])
        o.load_state_dict(checkpoint["optimizer_state_dict"])
