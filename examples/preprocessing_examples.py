import pandas as pd
import joblib
import os
import sys
from pathlib import Path
from typing import Dict

# IMPROVE general imports
from improvelib.initializer.stage_config import PreprocessConfig
from improvelib import utils # TODO consider "... import utils as imp_utils"

# DRP specific imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.applications.drug_response_prediction.omics_utils as omics
import improvelib.applications.drug_response_prediction.drug_utils as drug
import improvelib.applications.drug_response_prediction.drp_utils as drp

# LGBM specific imports
from lgbm_model_parameters import model_params
from LGBM.model_utils.utils import gene_selection, scale_df

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Required functions for all improve scripts are main() and run().
# main() initializes parameters and calls run() with the parameters.


def run(params: Dict, logger=None):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
        cfg (Preprocess): Preprocess object. Default is None. TODO currently logger is passed?!

    Returns:
        str: status of the preprocessing.
    """
    logger.info("Running preprocessing.") if logger else print(
        "Running preprocessing.")
    logger.debug(f"Loading data from {params['input_dir']}.")

    params = utils.build_paths(params)

    # Create output dir for model input data (to save preprocessed ML data)
    utils.create_outdir(outdir=params["ml_data_outdir"])
    print("\nLoads omics data.")
    omics_obj = omics.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv']  # return gene expression

    print("\nLoad drugs data.")
    drugs_obj = drug.DrugsLoader(params)
    md = drugs_obj.dfs['drug_mordred.tsv']  # return the Mordred descriptors
    md = md.reset_index()  # TODO. implement reset_index() inside the loader

    # Gene selection (based on LINCS landmark genes)
    if params["use_lincs"]:
        genes_fpath = filepath/"LGBM/model_utils/landmark_genes.txt"
        ge = gene_selection(
            ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # Prefix gene column names with "ge."
    fea_sep = "."
    fea_prefix = "ge"
    ge = ge.rename(
        columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})

    print("Create feature scaler.")
    rsp_tr = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

    # Retain feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge(ge[params["canc_col_name"]],
                    on=params["canc_col_name"], how="inner")
    rsp = rsp.merge(md[params["drug_col_name"]],
                    on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(
        rsp[params["canc_col_name"]])].reset_index(drop=True)
    md_sub = md[md[params["drug_col_name"]].isin(
        rsp[params["drug_col_name"]])].reset_index(drop=True)

    # Scale gene expression
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]
                           ) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    # Scale Mordred descriptors
    _, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    md_scaler_fpath = Path(params["ml_data_outdir"]
                           ) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]

        # --------------------------------
        # Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge(ge[params["canc_col_name"]],
                        on=params["canc_col_name"], how="inner")
        rsp = rsp.merge(md[params["drug_col_name"]],
                        on=params["drug_col_name"], how="inner")
        ge_sub = ge[ge[params["canc_col_name"]].isin(
            rsp[params["canc_col_name"]])].reset_index(drop=True)
        md_sub = md[md[params["drug_col_name"]].isin(
            rsp[params["drug_col_name"]])].reset_index(drop=True)

        # Scale features
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)  # scale gene expression
        # scale Mordred descriptors
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)
        # print("GE mean:", ge_sc.iloc[:,1:].mean(axis=0).mean())
        # print("GE var: ", ge_sc.iloc[:,1:].var(axis=0).mean())
        # print("MD mean:", md_sc.iloc[:,1:].mean(axis=0).mean())
        # print("MD var: ", md_sc.iloc[:,1:].var(axis=0).mean())

        # --------------------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step depends on the model.
        # --------------------------------
        # [Req] Build data name
        data_fname = utils.build_ml_data_name(params, stage)

        print("Merge data")
        data = rsp.merge(ge_sc, on=params["canc_col_name"], how="inner")
        data = data.merge(md_sc, on=params["drug_col_name"], how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True)  # shuffle

        print("Save data")
        # to_parquet() throws error since "study" contain mixed values
        data = data.drop(columns=["study"])
        # saves ML data file to parquet
        data.to_parquet(Path(params["ml_data_outdir"])/data_fname)

        # Prepare the y dataframe for the current stage
        fea_list = ["ge", "mordred"]
        fea_cols = [c for c in data.columns if (
            c.split(fea_sep)[0]) in fea_list]
        meta_cols = [c for c in data.columns if (
            c.split(fea_sep)[0]) not in fea_list]
        ydf = data[meta_cols]

        # [Req] Save y dataframe for the current stage
        utils.save_stage_ydf(ydf, params, stage)


# def example_parameter_initialization_1():
#     # List of custom parameters, if any
#     # can be list or file in json or yaml format, e.g. :
#     # my_additional_definitions = [{"name": "param_name", "type": str, "default": "default_value", "help": "help message"}]
#     # my_additional_definitions = None
#     # my_additional_definitions = Path("custom_params.json")
#     # my_additional_definitions = filepath/"custom_params.json"

#     # Set additional_definitions to None if no custom parameters are needed

#     # Exampple 1: Initialize parameters using the Preprocess class

#     # Initialize parameters using the Preprocess class


#     custom_config_options = []

#     cfg = PreprocessConfig()

#     params = cfg.initialize_parameters(filepath,
#                                        default_config="default.cfg",
#                                        additional_definitions=[],
#                                        required=None
#                                        )
#     return params, cfg.logger


def example_parameter_from_dictionary():
    # Example 2: Initialize parameters using custom list

    # Initialize parameters using custom parameters list defined in Python argparse.ArgumentParser format
    my_params_example = [
        # see argparse.ArgumentParser.add_argument() for more information
        {
            # name of the argument
            "name": "y_data_files",
            # type of the argument
            "type": str,
            # number of arguments that should be consumed
            "nargs": "+",
            # help message
            "help": "List of files that contain the y (prediction variable) data.",
        },
        {
            # name of the argument
            "name": "supplement",
            # type of the argument
            "type": str,
            # number of arguments that should be consumed
            "nargs": 2,
            # name of the argument in usage messages
            "metavar": ("FILE", "TYPE"),
            # action to be taken when this argument is encountered
            "action": "append",
            "help": "Supplemental data tuple FILE and TYPE. FILE is in INPUT_DIR.",   # help message
        }
    ]
    
    return my_params_example





# Required functions for all improve model scripts are main() and run().
# main() initializes parameters and calls run() with the parameters.
# run() is the function that executes the primary processing of the model script.
def main(args):
    # params1, logger1 = example_parameter_initialization_1()
    # params2, logger2 = example_parameter_initialization_2()

    model_cli_config_params = example_parameter_from_dictionary()
 

    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters('LGBM',
                                       default_config=None, # e.g. default.cfg or default.ini
                                       additional_cli_section='LGBM',
                                       additional_definitions=model_cli_config_params,
                                       required=None
                                       )

    status_lgbm = run(params_lgbm, logger_lgbm)
    
    cfg.logger.info(f"Preprocessing completed with {status_lgbm}. Data saved in {params['output_dir']}") if cfg.logger else print(
        f"Preprocessing completed. Data saved in {params['output_dir]}")
    


if __name__ == "__main__":
    main(sys.argv[1:])
