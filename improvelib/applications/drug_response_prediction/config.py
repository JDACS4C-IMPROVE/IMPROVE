"""
Configuration for drug response prediction (DRP) models.

This module defines configuration classes for preprocessing, training,
and inference of drug response prediction models in the IMPROVE framework.

Classes:
    DRPPreprocessConfig: Configuration for preprocessing drug response data.
    DRPTrainConfig: Configuration for training drug response models.
    DRPInferConfig: Configuration for inference with drug response models.
"""

from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig
from improvelib.applications.drug_response_prediction.drp_params_def import app_preproc_params, app_train_params, app_infer_params

class DRPPreprocessConfig(PreprocessConfig):
    """Configuration for preprocessing drug response data.

    This class extends the PreprocessConfig to include specific parameters
    for preprocessing in the context of monotherapy drug response prediction.

    Attributes:
        _preproc_params (list): List of dictionaries defining preprocessing parameters.
    """

    

    def __init__(self):
        """Initializes the DRPPreprocessConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=app_preproc_params,
            group='Drug Response Prediction Preprocessing'
        )


class DRPTrainConfig(TrainConfig):
    """Configuration for training drug response models.

    This class extends the TrainConfig to include specific parameters
    for training in the context of monotherapy drug response prediction.

    Attributes:
        _app_train_params (list): List of dictionaries defining training parameters.
    """



    def __init__(self):
        """Initializes the DRPTrainConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=app_train_params,
            group='Drug Response Prediction Training'
        )


class DRPInferConfig(InferConfig):
    """Configuration for inference with drug response models.

    This class extends the InferConfig to include specific parameters
    for inference in the context of monotherapy drug response prediction.

    Attributes:
        _app_infer_params (list): List of dictionaries defining inference parameters.
    """



    def __init__(self):
        """Initializes the DRPInferConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=app_infer_params,
            group='Drug Response Prediction Inference'
        )
