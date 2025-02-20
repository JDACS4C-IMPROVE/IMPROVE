model_scripts_dir = "./LGBM"
model_name = "lgbm"
lca_splits_dir = "./lc_splits"
#split = 0
dataset = "CCLE"
output_dir = "./snakemake_output"
input_dir = "./LGBM/csa_data/raw_data"
epochs = 3
split_nums = ['0', '1']
conda_env = "lgbm_py37"

(SHARDS,) = glob_wildcards(lca_splits_dir + "/" + dataset + "_split_" + split_nums[0] + "_sz_{shard}.txt")

def get_preprocess_params(wildcards):
    files = dict()
    # val and test don't change per shard
    files["val_split_file"] = dataset + "_split_" + wildcards.split + "_val.txt"
    files["test_split_file"] = dataset + "_split_" + wildcards.split + "_test.txt"
    # train and output_dir change per shard
    train_fname = dataset + "_split_0_sz_" + wildcards.shard + ".txt"
    files["train_split_file"] = str(os.path.join(lca_splits_dir, train_fname))
    files["output_dir"] = output_dir + "/ml_data/split_" + wildcards.split + "/sz_" + wildcards.shard
    return files

def get_train_params(wildcards):
    files = dict()
    files["input_dir"] = output_dir + "/ml_data/split_" + wildcards.split + "/sz_" + wildcards.shard
    files["output_dir"] = output_dir + "/models/split_" + wildcards.split + "/sz_" + wildcards.shard
    return files

def get_infer_params(wildcards):
    files = dict()
    files["input_data_dir"] = output_dir + "/ml_data/split_" + wildcards.split + "/sz_" + wildcards.shard
    files["input_model_dir"] = output_dir + "/models/split_" + wildcards.split + "/sz_" + wildcards.shard
    files["output_dir"] = output_dir + "/infer/split_" + wildcards.split + "/sz_" + wildcards.shard
    return files

rule all:
    input:
        expand(output_dir + "/infer/split_{split}/sz_{shard}/test_y_data_predicted.csv", shard=SHARDS, split=split_nums),

rule preprocess:
    input:
        lca_splits_dir + "/" + dataset + "_split_{split}_sz_{shard}.txt",
        input_dir + "/splits/" + dataset + "_split_{split}_val.txt",
        input_dir + "/splits/" + dataset + "_split_{split}_test.txt",
    output:
        output_dir + "/ml_data/split_{split}/sz_{shard}/test_y_data.csv"
    conda:
        conda_env
    params:
        get_preprocess_params
    shell:
        'mkdir -p logs/{rule}/split_{wildcards.split}; '
        'exec 2> logs/{rule}/split_{wildcards.split}/sz_{wildcards.shard}.e; '
        'exec 1> logs/{rule}/split_{wildcards.split}/sz_{wildcards.shard}.o; '
        'export PYTHONPATH=./IMPROVE; '
        "python {model_scripts_dir}/{model_name}_preprocess_improve.py --train_split_file {params[0][train_split_file]} --val_split_file {params[0][val_split_file]} --test_split_file {params[0][test_split_file]} --input_dir {input_dir} --output_dir {params[0][output_dir]}"

rule train:
    input:
        output_dir + "/ml_data/split_{split}/sz_{shard}/test_y_data.csv"
    output:
        output_dir + "/models/split_{split}/sz_{shard}/val_y_data_predicted.csv"
    conda:
        conda_env
    params:
        get_train_params
    shell:
        'mkdir -p logs/{rule}/split_{wildcards.split}; '
        'exec 2> logs/{rule}/split_{wildcards.split}/sz_{wildcards.shard}.e; '
        'exec 1> logs/{rule}/split_{wildcards.split}/sz_{wildcards.shard}.o; '
        'export PYTHONPATH=./IMPROVE; '
        "python {model_scripts_dir}/{model_name}_train_improve.py --input_dir {params[0][input_dir]} --output_dir {params[0][output_dir]} --epochs {epochs}"

rule infer:
    input:
        output_dir + "/models/split_{split}/sz_{shard}/val_y_data_predicted.csv"
    output:
        output_dir + "/infer/split_{split}/sz_{shard}/test_y_data_predicted.csv"
    conda:
        conda_env
    params:
        get_infer_params
    shell:
        'mkdir -p logs/{rule}/split_{wildcards.split}; '
        'exec 2> logs/{rule}/split_{wildcards.split}/sz_{wildcards.shard}.e; '
        'exec 1> logs/{rule}/split_{wildcards.split}/sz_{wildcards.shard}.o; '
        'source /data/koussanc/conda/etc/profile.d/conda.sh; '
        "python {model_scripts_dir}/{model_name}_infer_improve.py --input_data_dir {params[0][input_data_dir]} --input_model_dir {params[0][input_model_dir]} --output_dir {params[0][output_dir]} --calc_infer_scores true"
