"""
Configuration for synergy models.

This module defines configuration classes for preprocessing, training,
and inference of synergy models in the IMPROVE framework.

Classes:
    SynergyPreprocessConfig: Configuration for preprocessing synergy data.
    SynergyTrainConfig: Configuration for training synergy models.
    SynergyInferConfig: Configuration for inference with synergy models.
"""

from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig
from improvelib.applications.synergy.synergy_params_def import app_preproc_params, app_train_params, app_infer_params


class SynergyPreprocessConfig(PreprocessConfig):
    """Configuration for preprocessing synergy data.

    This class extends the PreprocessConfig to include specific parameters
    for preprocessing in the context of synergy prediction.

    Attributes:
        _preproc_params (list): List of dictionaries defining preprocessing parameters.
    """

    def __init__(self):
        """Initializes the SynergyPreprocessConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=app_preproc_params,
            group='Synergy Prediction Preprocessing'
        )


class SynergyTrainConfig(TrainConfig):
    """Configuration for training synergy models.

    This class extends the TrainConfig to include specific parameters
    for training in the context of synergy prediction.

    Attributes:
        _app_train_params (list): List of dictionaries defining training parameters.
    """

    def __init__(self):
        """Initializes the SynergyTrainConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=app_train_params,
            group='Synergy Prediction Training'
        )


class SynergyInferConfig(InferConfig):
    """Configuration for inference with synergy models.

    This class extends the InferConfig to include specific parameters
    for inference in the context of synergy prediction.

    Attributes:
        _app_infer_params (list): List of dictionaries defining inference parameters.
    """

    def __init__(self):
        """Initializes the SynergyInferConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=app_infer_params,
            group='Synergy Prediction Inference'
        )
