from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig


class DRPPreprocessConfig(PreprocessConfig):

    # App-specific params (App: monotherapy drug response prediction)
    # Note! This list should not be modified (i.e., no params should added or
    # removed from the list.
    #
    # There are two types of params in the list: default and required
    # default:   default values should be used
    # required:  these params must be specified for the model in the param file
    _preproc_params = [
        {"name": "y_data_files",  # default
         "type": str,
         "help": "List of files that contain the y (prediction variable) data. \
                Example: [['response.tsv']]",
         },
        {"name": "x_data_canc_files",  # required
         "type": str,
         "help": "List of feature files including gene_system_identifer. Examples: \n\
                1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
                2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
         },
        {"name": "x_data_drug_files",  # required
         "type": str,
         "help": "List of feature files. Examples: \n\
                1) [['drug_SMILES.tsv']] \n\
                2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
         },
        {"name": "canc_col_name",
         "default": "improve_sample_id",  # default
         "type": str,
         "help": "Column name in the y (response) data file that contains the cancer sample ids.",
         },
        {"name": "drug_col_name",  # default
         "default": "improve_chem_id",
         "type": str,
         "help": "Column name in the y (response) data file that contains the drug ids.",
         },
        {"name": "data_format",  # default
         "default": ".parquet",
         "type": str,
         "help": "Format to load and save data",
         },
    ]

    def __init__(self):
        super().__init__()
        self.cli.set_command_line_options(options=self._preproc_params,
                                          group='Drug Response Prediction Preprocessing')


class DRPTrainConfig(TrainConfig):
    _app_train_params = [
        {"name": "data_format",  # default
         "type": str,
         "help": "Format to load and save data",
         },
        {"name": "y_col_name",  # default
         "type": str,
         "help": "Name of the metric to predict",
         },
        {"name": "y_data_suffix",  # default
         "type": str,
         "help": "Suffix for the columns in prediction file",
         }
    ]

    def __init__(self):
        super().__init__()
        self.cli.set_command_line_options(
            options=self._app_train_params,
            group='Drug Response Prediction Training')


class DRPInferConfig(InferConfig):
    _app_infer_params = [
        {"name": "data_format",  # default
                 "type": str,
                 "help": "Format to load and save data",
         },
        {"name": "y_col_name",  # default
         "type": str,
         "help": "Name of the metric to predict",
         },
        {"name": "y_data_suffix",  # default
         "type": str,
         "help": "Suffix for the columns in prediction file",
         },
        {"name": "model_file_format",  # default
         "type": str,
         "help": "File format for the model",
         },
        {"name": "model_file_name",  # default
         "type": str,
         "help": "File name for the model",
         },
        {"name": "y_data_preds_suffix",  # default
         "type": str,
         "help": "???",
         },
        {"name": "loss",  # default
         "type": str,
         "help": "???",
         },
        {"name": "json_scores_suffix",  # default
         "type": str,
         "help": "???",
         }
    ]

    def __init__(self):
        super().__init__()
        self.cli.set_command_line_options(
            options=self._app_infer_params,
            group='Drug Response Prediction Inference')
