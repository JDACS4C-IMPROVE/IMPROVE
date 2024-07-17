""" Functionality for IMPROVE drug response prediction (DRP) models. """

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

from ast import literal_eval

import pandas as pd


# Set logger for this module

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))

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
                 sep: str = "\t",
                 verbose: bool = True):

        self.smiles_fname = "drug_SMILES.tsv"
        self.mordred_fname = "drug_mordred.tsv"
        self.ecfp4_512bit_fname = "drug_ecfp4_nbits512.tsv"
        self.known_file_names = [self.smiles_fname,
                                 self.mordred_fname,
                                 self.ecfp4_512bit_fname]

        self.params = params
        self.sep = sep

        if isinstance(params["x_data_drug_files"], str):
            # instanciate array from string
            self.inp = literal_eval(params["x_data_drug_files"])
        else:
            self.inp = params["x_data_drug_files"]

        logger.debug(f"self.inp: {self.inp}")

        self.drug_col_name = params["drug_col_name"]
        self.x_data_path = params["x_data_path"]
        self.dfs = {}
        self.verbose = verbose

        if self.verbose:
            # print(f"x_data_drug_files: {params['x_data_drug_files']}")
            print(f"drug_col_name: {params['drug_col_name']}")
            print("x_data_drug_files:")
            for i, d in enumerate(self.inp):
                print(f"{i+1}. {d}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(
                i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
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
        fpath = Path(fpath)
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_drug_data(self, fname):
        """ ... """
        fpath = os.path.join(self.x_data_path, fname)
        DrugsLoader.check_path(fpath)
        # if self.verbose:
        #     print(f"Loading {fpath}")
        df = pd.read_csv(fpath, sep=self.sep)
        df = df.set_index(self.drug_col_name)
        return df

    def load_all_drug_data(self):
        """ ... """
        for i in self.inp:
            fname = i[0]
            self.dfs[fname] = self.load_drug_data(fname)

        # print(self.dfs[self.smiles_fname].iloc[:4, :])
        # print(self.dfs[self.mordred_fname].iloc[:4, :4])
        # print(self.dfs[self.ecfp4_512bit_fname].iloc[:4, :4])
        print("Finished loading drug data.")