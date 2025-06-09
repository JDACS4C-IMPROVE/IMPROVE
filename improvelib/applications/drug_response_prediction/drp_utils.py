"""
This module provides utilities for loading and processing response data for 
drug response prediction models in the IMPROVE framework. It also provides 
functionality to filter dataframes, retaining only the common IDs shared between them.
"""

# Standard library imports
from ast import literal_eval
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from improvelib.utils_app_generic import get_x_data, get_response_data, get_all_response_data, determine_transform, transform_data, get_response_with_features, get_features_in_response
from improvelib.statics import L1000_ENTREZ, L1000_SYMBOL, LINCS_SYMBOL
from improvelib.applications.drug_response_prediction.drp_statics import methyl_symbol_dict, methyl_entrez_dict, methyl_ensembl_dict

# Set logger for this module
FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))



def change_gene_identifiers(data, data_type, identifier):
    if data_type == 'methyl' or data_type == 'methylation':
        if identifier == 'Symbol' or identifier == 'symbol' or identifier == 'gene_symbol':
            data = data.rename(columns=methyl_symbol_dict)
        elif identifier == 'Entrez' or identifier == 'entrez':
            data = data.rename(columns=methyl_entrez_dict)
        elif identifier == 'Ensembl' or identifier == 'ensembl':
            data = data.rename(columns=methyl_ensembl_dict)
        else:
            raise ValueError(f"ERROR! Identfied provided was {identifier} but must be one of 'Entrez' or 'Symbol' or 'Ensembl'.\n")
    else:
        print("Only methylation has been implemented.")
    return data



