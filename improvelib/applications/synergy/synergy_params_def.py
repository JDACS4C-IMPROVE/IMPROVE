app_preproc_params = [
    {
        "name": "y_data_files",
        "type": str,
        "default": "synergy.tsv",
        "help": (
            "File that contain the y (prediction variable) data. "
        ),
    },
    {
        "name": "cell_column_name",
        "type": str,
        "default": "DepMapID",
        "help": (
            "Column name in the y (response) data file that contains the cancer sample IDs."
        ),
    },
    {
        "name": "drug_1_column_name",
        "type": str,
        "default": "DrugID_row",
        "help": (
            "Column name in the y (response) data file that contains the first drug IDs."
        ),
    },
    {
        "name": "drug_2_column_name",
        "type": str,
        "default": "DrugID_col",
        "help": (
            "Column name in the y (response) data file that contains the second drug IDs."
        ),
    },
    {
        "name": "y_col_name",
        "type": str,
        "default": "loewe",
        "help": (
            "Column name in the y data file (e.g., synergy.tsv), that represents "
            "the target variable that the model predicts. In synergy prediction "
            "problem it can be one of ['loewe', 'bliss', 'zip', 'hsa', 'smean', 'css']."
        ),
    },
]

app_train_params = [
    {
        "name": "y_col_name",
        "type": str,
        "default": "loewe",
        "help": (
            "Column name in the y data file (e.g., synergy.tsv), that represents "
            "the target variable that the model predicts. In synergy prediction "
            "problem it can be one of ['loewe', 'bliss', 'zip', 'hsa', 'smean', 'css']."
        ),
    },
]

app_infer_params = [
    {
        "name": "y_col_name",
        "type": str,
        "default": "loewe",
        "help": (
            "Column name in the y data file (e.g., response.tsv), that represents "
            "the target variable that the model predicts. In synergy prediction "
            "problem it can be one of ['loewe', 'bliss', 'zip', 'hsa', 'smean', 'css']."
        ),
    },
]