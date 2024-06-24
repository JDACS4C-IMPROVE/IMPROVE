
""" Functionality for IMPROVE drug response prediction (DRP) models. """

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import sys
import argparse

from ast import literal_eval

import pandas as pd

# Set logger for this module

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.ERROR))


from improvelib import framework as frm

class FooAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)
        
drugs_params = [
    {"name": "smiles_fname",
     "default":"drug_SMILES.tsv",
     "type": str,
     "help": "SMILES filename",
    },
    {"name": "mordred_fname",
     "default":"drug_mordred.tsv",
     "action": FooAction,
     "type": str,
     "help": "Mordred filename",
    },
    {"name": "ecfp4_512bit_fname",
     "default": "drug_ecfp4_nbits512.tsv",
     "type": str,
     "help": "Extended Connectivity Fingerprint, up to four bonds (ECFP4) 512 bit filename", # TODO options for different bits?
    },
]

# ==============================================================
# Drug feature loaders
# ==============================================================

class DrugsLoader():
    """
    Example:
        from improve import drug_resp_pred as drp
        drugs_loader = drp.DrugsLoader(params)
        print(drugs_loader)
        print(dir(drugs_loader))
        smi = drugs_loader["drug_SMILES.tsv"]
    """
    def __init__(self,
                 params: Dict,  # improve params
                 benchmark: Dict,
                 sep: str = "\t",
                 verbose: bool = True):

        self.smiles_fname = "drug_SMILES.tsv"
        self.mordred_fname = params["mordred_fname"]
        self.ecfp4_512bit_fname = "drug_ecfp4_nbits512.tsv"
        self.known_file_names = [self.smiles_fname,
                                 self.mordred_fname,
                                 self.ecfp4_512bit_fname]

        self.params = params
        self.sep = sep
        self.benchmark = benchmark
        
        if isinstance(self.params["x_data_drug_files"], str):
            # instantiate array from string
            self.inp = literal_eval(self.params["x_data_drug_files"])
        else:
            self.inp = self.params["x_data_drug_files"]
            
        self.drug_col_name = params["drug_col_name"]
        self.x_data_path = params["x_data_path"]
        self.dfs = {}
        self.verbose = verbose
 
        print("File:")
        if hasattr(benchmark.parser.parse_args(), 'mordred_fname_nondefault'):
            print(benchmark.parser.parse_args().mordred_fname)
            self.x_data_path = None
        
        logger.debug(f"self.inp: {self.inp}")

        if self.verbose:
            # print(f"x_data_drug_files: {params['x_data_drug_files']}")
            print(f"drug_col_name: {params['drug_col_name']}")
            print("x_data_drug_files:")
            for i, d in enumerate(self.inp):
                print(f"{i+1}. {d}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])
        # print(self.inp_fnames)

        self.load_all_drug_data()

    def __repr__(self):
        if self.dfs:
            return "Loaded data:\n" + "\n".join([f"{fname}: {df.shape}" for fname, df in self.dfs.items()])
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath):
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_drug_data(self, fname):
        """ ... """
        if self.x_data_path is None:
            fpath = Path(os.path.abspath(fname))
        else:
            fpath = self.x_data_path / fname
        DrugsLoader.check_path(fpath)
        if self.verbose:
            print(f"Loading {fpath}")
        df = pd.read_csv(fpath, sep=self.sep)
        df = df.set_index(self.drug_col_name)
        return df

    def load_all_drug_data(self):
        """ ... """
        for i in self.inp:
            #fname = i[0]
            fname = self.params[i[0]]
            self.dfs[fname] = self.load_drug_data(fname)

        # print(self.dfs[self.smiles_fname].iloc[:4, :])
        # print(self.dfs[self.mordred_fname].iloc[:4, :4])
        # print(self.dfs[self.ecfp4_512bit_fname].iloc[:4, :4])
        print("Finished loading drug data.")