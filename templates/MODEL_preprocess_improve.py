""" Preprocess benchmark data (e.g., CSA data) to generate datasets for the prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["output_dir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].

2. Y data files.
   The script creates dataframes with true y values and additional metadata.
   Generated files:
        train_y_data.csv, val_y_data.csv, and test_y_data.csv.
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE imports
# Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.utils as frm
# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specifc imports
from model_params_def import preprocess_params # [Req]
### Insert other required imports here

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated) ML data files.
    """


    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    # not needed for this data/model

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    
    """ Here is an example of loading gene expression data
    print("Load omics data.")
    ge = drp.get_x_data(file = params['cell_transcriptomic_file'], 
                        benchmark_dir = params['input_dir'], 
                        column_name = params['canc_col_name'])


    print("Load drug data.")
    md = drp.get_x_data(file = params['drug_mordred_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])

    print("Load train response data.")
    response_train = drp.get_response_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   response_file=params['y_data_file'])
    """
    print("Find intersection of training data.")
    response_train = drp.get_response_with_features(response_train, ge, params['canc_col_name'])
    response_train = drp.get_response_with_features(response_train, md, params['drug_col_name'])
    ge_train = drp.get_features_in_response(ge, response_train, params['canc_col_name'])
    md_train = drp.get_features_in_response(md, response_train, params['drug_col_name'])

    print("Determine transformations.")
    drp.determine_transform(ge_train, 'ge_transform', params['cell_transcriptomic_transform'], params['output_dir'])
    drp.determine_transform(md_train, 'md_transform', params['drug_mordred_transform'], params['output_dir'])

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = drp.get_response_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                response_file=params['y_data_file'])
        response_stage = drp.get_response_with_features(response_stage, ge, params['canc_col_name'])
        response_stage = drp.get_response_with_features(response_stage, md, params['drug_col_name'])
        ge_stage = drp.get_features_in_response(ge, response_stage, params['canc_col_name'])
        md_stage = drp.get_features_in_response(md, response_stage, params['drug_col_name'])

        print(f"Transform {stage} data.")
        ge_stage = drp.transform_data(ge_stage, 'ge_transform', params['output_dir'])
        md_stage = drp.transform_data(md_stage, 'md_transform', params['output_dir'])

        # Prefix gene column names with "ge."
        fea_sep = "."
        fea_prefix = "ge"
        ge_stage = ge_stage.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge_stage.columns[1:]})

        # [Req] Build data name
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

        print(f"Merge {stage} data")
        data = response_stage.drop(columns=["study"]) # to_parquet() throws error since "study" contain mixed values
        y_df_cols = data.columns.tolist()
        data = data.merge(ge_stage, on=params["canc_col_name"], how="inner")
        data = data.merge(md_stage, on=params["drug_col_name"], how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print(f"Save {stage} data")
        
        data.to_parquet(Path(params["output_dir"]) / data_fname) # saves ML data file to parquet
        
        # [Req] Save y dataframe for the current stage
        ydf = data[y_df_cols]
        frm.save_stage_ydf(ydf, stage, params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="lgbm_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
