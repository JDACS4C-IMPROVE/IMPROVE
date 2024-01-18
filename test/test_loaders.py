import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

filepath = Path(__file__).resolve().parent

app_preproc_params = [
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # required
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # required
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id", # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {"name": "drug_col_name", # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
    },

]

additional_definitions = app_preproc_params
params = frm.initialize_parameters(
    filepath,
    # default_model="test_params.txt",
    default_model="test_params_all.txt",
    additional_definitions=additional_definitions,
    required=None,
)

params = frm.build_paths(params)

# breakpoint()
print("\nLoad omics data.\n")
omics_obj = drp.OmicsLoader(params)
print(omics_obj)
copy_number = omics_obj.dfs['cancer_copy_number.tsv']
disc_copy_number = omics_obj.dfs['cancer_discretized_copy_number.tsv']
dna_methyl = omics_obj.dfs['cancer_DNA_methylation.tsv']
gene_exp = omics_obj.dfs['cancer_gene_expression.tsv']
mirna = omics_obj.dfs['cancer_miRNA_expression.tsv']
mut_cnt = omics_obj.dfs['cancer_mutation_count.tsv']
mut_long_format = omics_obj.dfs['cancer_mutation_long_format.tsv']
mut = omics_obj.dfs['cancer_mutation.parquet']
rppa = omics_obj.dfs['cancer_RPPA.tsv']

# breakpoint()
print("\nLoad drugs data.\n")
drugs_obj = drp.DrugsLoader(params)
print(drugs_obj)
smi = drugs_obj.dfs['drug_SMILES.tsv']
fps = drugs_obj.dfs['drug_ecfp4_nbits512.tsv']
mrd = drugs_obj.dfs['drug_mordred.tsv']

# breakpoint()
print("\nLoad response data.\n")
rsp_tr = drp.DrugResponseLoader(params,
                                split_file=params["train_split_file"],
                                verbose=False).dfs["response.tsv"]
rsp_vl = drp.DrugResponseLoader(params,
                                split_file=params["val_split_file"],
                                verbose=False).dfs["response.tsv"]
rsp_te = drp.DrugResponseLoader(params,
                                split_file=params["test_split_file"],
                                verbose=False).dfs["response.tsv"]

# breakpoint()
print("\nFinished test.")
