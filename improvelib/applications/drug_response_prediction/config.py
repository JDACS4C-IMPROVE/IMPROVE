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


class DRPPreprocessConfig(PreprocessConfig):
    """Configuration for preprocessing drug response data.

    This class extends the PreprocessConfig to include specific parameters
    for preprocessing in the context of monotherapy drug response prediction.

    Attributes:
        _preproc_params (list): List of dictionaries defining preprocessing parameters.
    """

    _preproc_params = [
        {
            "name": "y_data_files",
            "type": str,
            "default": "[['response.tsv']]",
            "help": (
                "List of files that contain the y (prediction variable) data. "
                "Example: [['response.tsv']]"
            ),
        },
        {
            "name": "x_data_canc_files",
            "type": str,
            "default": "[['cancer_gene_expression.tsv', ['Gene_Symbol']]]",
            "help": (
                "List of files containing omics-related features and the identifiers to use. Examples:\n"
                "1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]]\n"
                "2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]]."
            ),
        },
        {
            "name": "x_data_drug_files",
            "type": str,
            "default": "[['drug_SMILES.tsv']]",
            "help": (
                "List of files containing drug-related features. Examples:\n"
                "1) [['drug_SMILES.tsv']]\n"
                "2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]"
            ),
        },
        {
            "name": "canc_col_name",
            "type": str,
            "default": "improve_sample_id",
            "help": (
                "Column name in the y (response) data file that contains the cancer sample IDs."
            ),
        },
        {
            "name": "drug_col_name",
            "type": str,
            "default": "improve_chem_id",
            "help": (
                "Column name in the y (response) data file that contains the drug IDs."
            ),
        },
        {
            "name": "y_col_name",
            "type": str,
            "default": "auc",
            "help": (
                "Column name in the y data file (e.g., response.tsv), that represents "
                "the target variable that the model predicts. In drug response prediction "
                "problem it can be IC50, AUC, and others."
            ),
        },
    ]

    def __init__(self):
        """Initializes the DRPPreprocessConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=self._preproc_params,
            group='Drug Response Prediction Preprocessing'
        )


class DRPTrainConfig(TrainConfig):
    """Configuration for training drug response models.

    This class extends the TrainConfig to include specific parameters
    for training in the context of monotherapy drug response prediction.

    Attributes:
        _app_train_params (list): List of dictionaries defining training parameters.
    """

    _app_train_params = [
        {
            "name": "y_col_name",
            "type": str,
            "default": "auc",
            "help": (
                "Column name in the y data file (e.g., response.tsv), that represents "
                "the target variable that the model predicts. In drug response prediction "
                "problem it can be IC50, AUC, and others."
            ),
        },
    ]

    def __init__(self):
        """Initializes the DRPTrainConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=self._app_train_params,
            group='Drug Response Prediction Training'
        )


class DRPInferConfig(InferConfig):
    """Configuration for inference with drug response models.

    This class extends the InferConfig to include specific parameters
    for inference in the context of monotherapy drug response prediction.

    Attributes:
        _app_infer_params (list): List of dictionaries defining inference parameters.
    """

    _app_infer_params = [
        {
            "name": "y_col_name",
            "type": str,
            "default": "auc",
            "help": (
                "Column name in the y data file (e.g., response.tsv), that represents "
                "the target variable that the model predicts. In drug response prediction "
                "problem it can be IC50, AUC, and others."
            ),
        },
    ]

    def __init__(self):
        """Initializes the DRPInferConfig."""
        super().__init__()
        self.cli.set_command_line_options(
            options=self._app_infer_params,
            group='Drug Response Prediction Inference'
        )
