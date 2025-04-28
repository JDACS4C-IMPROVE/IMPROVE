app_preproc_params = [
    {
        "name": "y_data_file",
        "type": str,
        "default": "synergy.tsv",
        "help": "File that contain the y (prediction variable) data."
    },
    {
        "name": "cell_transcriptomic_file",
        "type": str,
        "default": None,
        "help": "'cell_transcriptomics.tsv' for benchmark data or path to the transcriptomics data. None if not used."
    },
    {
        "name": "cell_transcriptomic_transform",
        "type": str,
        "default": None,
        "help": (
            "List of lists with the type of transformation and the option. "
            "Transformations will be performed in the order of the outer list. "
            "For example, [['normalize', 'zscale'], ['subset', 'L1000']] will first"
            "normalize the data with z-scaling and then subset to genes in L1000."
            "For subset, a path to a text file with Entrez IDs separated by new lines can be given."
            "None if not used."
        ),
    },
    {
        "name": "cell_cnv_file",
        "type": str,
        "default": None,
        "help": "'cell_cnv_continuous.tsv' or 'cell_cnv_discretized.tsv' for benchmark data or path to the CNV data. None if not used."
    },
    {
        "name": "cell_cnv_transform",
        "type": str,
        "default": None,
        "help": (
            "List of lists with the type of transformation and the option. "
            "Transformations will be performed in the order of the outer list. "
            "For example, [['normalize', 'zscale'], ['subset', 'L1000']] will first"
            "normalize the data with z-scaling and then subset to genes in L1000."
            "For subset, a path to a text file with Entrez IDs separated by new lines can be given."
            "None if not used."
        ),
    },
    {
        "name": "cell_mutation_file",
        "type": str,
        "default": None,
        "help": "'cell_mutation_delet.tsv' or 'cell_mutation_nonsynon.tsv' for benchmark data or path to the mutation data. None if not used."
    },
        {
        "name": "cell_mutation_transform",
        "type": str,
        "default": None,
        "help": (
            "List of lists with the type of transformation and the option. "
            "Transformations will be performed in the order of the outer list. "
            "For example, [['normalize', 'zscale'], ['subset', 'L1000']] will first"
            "normalize the data with z-scaling and then subset to genes in L1000."
            "For subset, a path to a text file with Entrez IDs separated by new lines can be given."
            "None if not used."
        ),
    },
    {
        "name": "drug_smiles_file",
        "type": str,
        "default": None,
        "help": "'drug_smiles.tsv' or 'drug_smiles_canonical.tsv' for benchmark data or path to the SMILES data. None if not used."
    },
    {
        "name": "drug_mordred_file",
        "type": str,
        "default": None,
        "help": "'drug_mordred.tsv' for benchmark data or path to the Mordred data. None if not used."
    },
    {
        "name": "drug_infomax_file",
        "type": str,
        "default": None,
        "help": "'drug_infomax.tsv' for benchmark data or path to the Infomax data. None if not used."
    },
    {
        "name": "drug_ecfp_file",
        "type": str,
        "default": None,
        "help": "'drug_ecfp[2/4/6]_nbits[256/1024].tsv' for benchmark data or path to the ECFP data. None if not used."
    },
    {
        "name": "cell_column_name",
        "type": str,
        "default": "DepMapID",
        "help": "Column name in the y (response) data file that contains the cancer sample IDs."
    },
    {
        "name": "drug_column_name",
        "type": str,
        "default": "DrugID",
        "help": "Column name in the y (response) data file that contains the cancer sample IDs."
    },
    {
        "name": "drug_1_column_name",
        "type": str,
        "default": "DrugID_row",
        "help": "Column name in the y (response) data file that contains the first drug IDs."
    },
    {
        "name": "drug_2_column_name",
        "type": str,
        "default": "DrugID_col",
        "help": "Column name in the y (response) data file that contains the second drug IDs."
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