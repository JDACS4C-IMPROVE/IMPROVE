# import framework and specific application 
from improve_lib import framework as frm
from improve_lib import drug_resp_pred as drp

# import model

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Note! This list should not be modified (i.e., no params should added or
# removed from the list.
#
# There are two types of params in the list: default and required
# default:   default values should be used
# required:  these params must be specified for the model in the param file
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

# 2. Model-specific params (Model: LightGBM)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
{"name": "use_lincs",
    "type": frm.str2bool,
    "default": True,
    "help": "Flag to indicate if landmark genes are used for gene selection.",
},
{"name": "scaling",
    "type": str,
    "default": "std",
    "choice": ["std", "minmax", "miabs", "robust"],
    "help": "Scaler for gene expression and Mordred descriptors data.",
},
{"name": "ge_scaler_fname",
    "type": str,
    "default": "x_data_gene_expression_scaler.gz",
    "help": "File name to save the gene expression scaler object.",
},
{"name": "md_scaler_fname",
    "type": str,
    "default": "x_data_mordred_scaler.gz",
    "help": "File name to save the Mordred scaler object.",
},
]

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------

# [Req]
def run(params: Dict):
""" Run data preprocessing.

Args:
    params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

Returns:
    str: directory name that was used to save the preprocessed (generated)
        ML data files.
"""

# ------------------------------------------------------
# [Req] Build paths and create output dir
# ------------------------------------------------------
# Build paths for raw_data, x_data, y_data, splits
params = frm.build_paths(params)

# Create output dir for model input data (to save preprocessed ML data)
frm.create_outdir(outdir=params["ml_data_outdir"])

# ------------------------------------------------------
# [Req] Load X data (feature representations)
# ------------------------------------------------------
# Use the provided data loaders to load data that is required by the model.
#
# Benchmark data includes three dirs: x_data, y_data, splits.
# The x_data contains files that represent feature information such as
# cancer representation (e.g., omics) and drug representation (e.g., SMILES).
#
# Prediction models utilize various types of feature representations.
# Drug response prediction (DRP) models generally use omics and drug features.
#
# If the model uses omics data types that are provided as part of the benchmark
# data, then the model must use the provided data loaders to load the data files
# from the x_data dir.
print("\nLoads omics data.")
omics_obj = drp.OmicsLoader(params)
# print(omics_obj)
ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression

print("\nLoad drugs data.")
drugs_obj = drp.DrugsLoader(params)
# print(drugs_obj)
md = drugs_obj.dfs['drug_mordred.tsv'] # return the Mordred descriptors
md = md.reset_index()  # TODO. implement reset_index() inside the loader

# ------------------------------------------------------
# Further preprocess X data
# ------------------------------------------------------
# Gene selection (based on LINCS landmark genes)
if params["use_lincs"]:
    genes_fpath = filepath/"landmark_genes"
    ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

# Prefix gene column names with "ge."
fea_sep = "."
fea_prefix = "ge"
ge = ge.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})

# ------------------------------------------------------
# Create feature scaler
# ------------------------------------------------------
# Load and combine responses
print("Create feature scaler.")
rsp_tr = drp.DrugResponseLoader(params,
                                split_file=params["train_split_file"],
                                verbose=False).dfs["response.tsv"]
rsp_vl = drp.DrugResponseLoader(params,
                                split_file=params["val_split_file"],
                                verbose=False).dfs["response.tsv"]
rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

# Retian feature rows that are present in the y data (response dataframe)
# Intersection of omics features, drug features, and responses
rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
rsp = rsp.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
md_sub = md[md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

# Scale gene expression
_, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
joblib.dump(ge_scaler, ge_scaler_fpath)
print("Scaler object for gene expression: ", ge_scaler_fpath)

# Scale Mordred descriptors
_, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
joblib.dump(md_scaler, md_scaler_fpath)
print("Scaler object for Mordred:         ", md_scaler_fpath)

del rsp, rsp_tr, rsp_vl, ge_sub, md_sub

# ------------------------------------------------------
# [Req] Construct ML data for every stage (train, val, test)
# ------------------------------------------------------
# All models must load response data (y data) using DrugResponseLoader().
# Below, we iterate over the 3 split files (train, val, test) and load
# response data, filtered by the split ids from the split files.

# Dict with split files corresponding to the three sets (train, val, and test)
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
    rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
    rsp = rsp.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
    md_sub = md[md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

    # Scale features
    ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler) # scale gene expression
    md_sc, _ = scale_df(md_sub, scaler=md_scaler) # scale Mordred descriptors

    # --------------------------------
    # [Req] Save ML data files in params["ml_data_outdir"]
    # The implementation of this step, depends on the model.
    # --------------------------------
    # [Req] Build data name
    data_fname = frm.build_ml_data_name(params, stage)

    print("Merge data")
    data = rsp.merge(ge_sc, on=params["canc_col_name"], how="inner")
    data = data.merge(md_sc, on=params["drug_col_name"], how="inner")
    data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

    print("Save data")
    data = data.drop(columns=["study"]) # to_parquet() throws error since "study" contain mixed values
    data.to_parquet(Path(params["ml_data_outdir"])/data_fname) # saves ML data file to parquet

    # Prepare the y dataframe for the current stage
    fea_list = ["ge", "mordred"]
    fea_cols = [c for c in data.columns if (c.split(fea_sep)[0]) in fea_list]
    meta_cols = [c for c in data.columns if (c.split(fea_sep)[0]) not in fea_list]
    ydf = data[meta_cols]

    # [Req] Save y dataframe for the current stage
    frm.save_stage_ydf(ydf, params, stage)

return params["ml_data_outdir"]

# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
