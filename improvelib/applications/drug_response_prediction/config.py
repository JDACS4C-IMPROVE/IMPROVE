"""
The help section strings of certain parameters is prepended with the following brackets,
encoding a suggested deprecation.
"[Dep+]" -- parameters we should try to deprecate in the upcoming release
"[Dep?]" -- parameters we might want to deprecate in the current or future release
"[Dep]"  -- parameters we plan to keep in the upcoming release, but deprecate in the future
"""

from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig


class DRPPreprocessConfig(PreprocessConfig):

    # App-specific params (App: monotherapy drug response prediction)
    #
    # There are two types of params in the list: default and required
    # default:   default values should be used
    # required:  these params must be specified for the model in the param file
    _preproc_params = [
        {"name": "y_data_files", # default expected
        "default": "fake", #NCK, not sure why we need this but we need a default
         "type": str,
         "help": "List of files that contain the y (prediction variable) data. \
                Example: [['response.tsv']]",
         },
        {"name": "x_data_canc_files", # required
         "default": "fake", #NCK, not sure why we need this but we need a default
         "type": str,
         "help": "List of feature files including gene_system_identifer. Examples: \n\
                1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
                2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
         },
        {"name": "x_data_drug_files", # required
        "default": "fake", #NCK, not sure why we need this but we need a default
         "type": str,
         "help": "List of feature files. Examples: \n\
                1) [['drug_SMILES.tsv']] \n\
                2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
         },
        {"name": "canc_col_name",
         "default": "improve_sample_id", # default expected
         "type": str,
         "help": "Column name in the y (response) data file that contains the cancer sample ids.",
         },
        {"name": "drug_col_name", # default expected
         "default": "improve_chem_id",
         "type": str,
         "help": "Column name in the y (response) data file that contains the drug ids.",
         },
        # ---------------------------------------
        {"name": "y_col_name", # Note! moved from improvelib param defs
         "type": str,
         "default": "auc",
         "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
         }, 
        # ---------------------------------------
        # TODO y_data_suffix is currently used in utils.py (previously framework.py)
        # We plan to hard-code this in the future release and deprecate. 
        {"name": "y_data_suffix", # default expected
         "type": str,
         "default": "y_data", # TODO Oleksandr: "gives error when default is not set"
         "help": "[Dep] Suffix to compose file name for storing true y dataframe."
         },

    ]

    def __init__(self):
        super().__init__()
        self.cli.set_command_line_options(
            options=self._preproc_params,
            group='Drug Response Prediction Preprocessing')


class DRPTrainConfig(TrainConfig):
    _app_train_params = [
        {"name": "y_col_name", # Note! moved from improvelib param defs
         "type": str,
         "default": "auc",
         "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
         },
        # ---------------------------------------
        # TODO y_data_suffix is currently used in utils.py (previously framework.py)
        # We plan to hard-code this in the future release and deprecate. 
        {"name": "y_data_suffix", # default expected
         "type": str,
         "default": "y_data", # TODO Oleksandr: "gives error when default is not set"
         "help": "[Dep] Suffix to compose file name for storing true y dataframe."
         },

    ]

    def __init__(self):
        super().__init__()
        self.cli.set_command_line_options(
             options=self._app_train_params,
             group='Drug Response Prediction Training')


class DRPInferConfig(InferConfig):
    _app_infer_params = [
        {"name": "y_col_name", # Note! moved from improvelib param defs
         "type": str,
         "default": "auc",
         "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
         },
        # ---------------------------------------
        # TODO y_data_suffix is currently used in utils.py (previously framework.py)
        # We plan to hard-code this in the future release and deprecate. 
        {"name": "y_data_suffix", # default expected
         "type": str,
         "default": "y_data", # TODO Oleksandr: "gives error when default is not set"
         "help": "[Dep] Suffix to compose file name for storing true y dataframe."
         },

    ]

    def __init__(self):
        super().__init__()
        self.cli.set_command_line_options(
            options=self._app_infer_params,
            group='Drug Response Prediction Inference')
