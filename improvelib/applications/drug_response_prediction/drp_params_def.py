app_preproc_params = [
    {
        "name": "y_data_file",
        "type": str,
        "default": "response.tsv",
        "help": "File that contain the y (prediction variable) data."
    },
    {
        "name": "cell_transcriptomic_file",
        "type": str,
        "default": None,
        "help": "'cancer_gene_expression.tsv' for benchmark data or path to the transcriptomics data. None if not used."
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
        "help": "'cancer_copy_number.tsv' or 'cancer_discretized_copy_number.tsv' for benchmark data or path to the CNV data. None if not used."
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
        "help": "'cancer_mutation_count.tsv' for benchmark data or path to the mutation data. None if not used."
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
        "name": "cell_methylation_file",
        "type": str,
        "default": None,
        "help": "'cancer_DNA_methylation.tsv' for benchmark data or path to the mutation data. None if not used."
    },
        {
        "name": "cell_methylation_transform",
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
        "name": "cell_miRNA_file",
        "type": str,
        "default": None,
        "help": "'cancer_miRNA_expression.tsv' for benchmark data or path to the mutation data. None if not used."
    },
        {
        "name": "cell_miRNA_transform",
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
        "name": "cell_RPPA_file",
        "type": str,
        "default": None,
        "help": "'cancer_RPPA.tsv' for benchmark data or path to the mutation data. None if not used."
    },
        {
        "name": "cell_RPPA_transform",
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
        "help": "'drug_SMILES.tsv' or 'drug_smiles_canonical.tsv' for benchmark data or path to the SMILES data. None if not used."
    },
    {
        "name": "drug_mordred_file",
        "type": str,
        "default": None,
        "help": "'drug_mordred.tsv' for benchmark data or path to the Mordred data. None if not used."
    },
    {
        "name": "drug_mordred_transform",
        "type": str,
        "default": None,
        "help": (
            "List of lists with the type of transformation and the option. "
            "None if not used."
        ),
    },
    {
        "name": "drug_ecfp_file",
        "type": str,
        "default": None,
        "help": "'drug_ecfp4_nbits512.tsv' for benchmark data or path to the ECFP data. None if not used."
    },
    {
        "name": "drug_ecfp_transform",
        "type": str,
        "default": None,
        "help": (
            "List of lists with the type of transformation and the option. "
            "None if not used."
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


app_train_params = [
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
]


app_infer_params = [
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
]
