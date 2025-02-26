# Synergy Benchmark Data Pipeline

## Overview

This repository contains scripts to generate benchmark data for the synergy application.

## Requirements

* python 3.12.2
* joblib 1.2.0
* numpy 1.26.0
* pandas 2.1.1
* rdkit 2023.09.2
* mordred 1.2.0
* dgl 2.1.0
* dgllife 0.3.2
* torch 2.2.1

## Installation and Setup

You can install the required packages using Conda and `benchmark_data.yml`:

```bash
conda env create -f benchmark_data.yml
```

Input data can be found from the following sources or [here] **ADD FTP.
* `CCLE_expression.csv`, `CCLE_gene_cn.csv`, `CCLE_mutations.csv`, and `sample_info.csv` are DepMap 22Q2 data and can be found [here](https://depmap.org/portal/download/all/).
* `drugcomb_summary_v_1_5.csv` can be found [here](https://drugcomb.org/download/).
* `drugcomb_drugs_df.csv` was generated with `synergy_data_drugs_todf.py` in the supplemental directory and the original data is [here](https://api.drugcomb.org/drugs).
* `checkedsmi_curated.csv` and `drugcomb_BIG_notindepmap_curated.csv` were hand-curated and are in the supplemental directory.


## Parameter Configuration

This workflow uses command line parameters are as follows:

* `--input_dir`: Location of the raw input data (default: './').
* `--output_dir`: Location to save the processed benchmark data (default: './').



## Usage

Run `pipeline.py` to generate synergy benchmark data:

```bash
python pipeline.py --input_dir </input_dir/> --output_dir </output_dir/>

```


## Output

* cell_cnv_continuous.tsv 
* cell_cnv_discretized.tsv  
* cell_mutation_delet.tsv 
* cell_mutation_nonsynon.tsv
* cell_transcriptomics.tsv
* drug_smiles.tsv
* drug_smiles_canonical.tsv
* drug_smiles_bad.tsv
* drug_mordred.tsv
* drug_infomax.tsv
* drug_ecfp2_nbits256.tsv	
* drug_ecfp4_nbits256.tsv	
* drug_ecfp6_nbits256.tsv 
* drug_ecfp2_nbits1024.tsv 
* drug_ecfp4_nbits1024.tsv
* drug_ecfp6_nbits1024.tsv
* synergy.tsv