"""
Unit tests for the bruteforce_csa workflow. 

"""

import unittest
import sys
import os

# Importing params definitions:
from csa_bruteforce_params_def import csa_bruteforce_params
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from pathlib import Path
filepath = Path(__file__).resolve().parent

csa_bruteforce_params = csa_bruteforce_params
config_file = 'csa_bruteforce_params.ini'

# Load config csa_bruteforce_params.ini
cfg = DRPPreprocessConfig()
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_bruteforce_params.ini",
    additional_definitions=csa_bruteforce_params,
    required=None
)

class Bruteforce_CSA_test(unittest.TestCase):
    def test_params(self):
        self.assertEqual(params['cuda_name'], 'cuda:0')
        self.assertEqual(params['csa_outdir'], './run_csa_full')
        self.assertEqual(params['source_datasets'], ['CCLE', 'gCSI'])
        self.assertEqual(params['target_datasets'], ['CCLE', 'gCSI'])
        self.assertEqual(params['split_nums'], ["0","1","2","3"])
        self.assertEqual(params['only_cross_study'], False)
        self.assertEqual(params['model_name'], 'GraphDRP')
        self.assertEqual(params['epochs'], 5)
        self.assertEqual(params['uses_cuda_name'], True)
    
        
if __name__ == '__main__':
    unittest.main()