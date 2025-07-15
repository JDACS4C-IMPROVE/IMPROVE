# drp benchmark updates with NCI-60 data
# note: NCI-60 data uses original drp cell data and it's own SMILES data
# drp splits for other studies are kept, new NCI-60 splits are generated
# all other drug data has been regenerated from SMILES

import os
import sys
import pandas as pd
import argparse
from pathlib import Path

from benchmark_data.synergy.chem_utils import canonicalize_smiles, generate_fingerprints, generate_mordred
from benchmark_data.splits.splits_generator import generate_mixed_splits, generate_blind_splits



def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate benchmark synergy data.')
    parser.add_argument('--input_dir',
                        type=str,
                        default='./',
                        help=f'Input directory with unprocessed data.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./',
                        help=f'Output directory to save processed benchmark data.')
    args = parser.parse_args(args)
    return args

def check_valid_NaNs(df):
    print(df.isna().sum().sum())
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(df.isna().sum().sum())
    return df

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csa_x_path = input_dir / "x_data"
    csa_y_path = input_dir / "y_data"
    csa_splits_path = input_dir / "splits"
    output_x_path = output_dir / "x_data"
    output_y_path = output_dir / "y_data"
    output_splits_path = output_dir / "splits"
    os.makedirs(output_x_path)
    os.makedirs(output_y_path)
    os.makedirs(output_splits_path)

    #################################### CELL DATA #########################################
    # removes triple header, ensures valid NaN values
    # no addition for NCI-60 data (only 11 cell lines are missing)
    cancer_copy_number = pd.read_csv(csa_x_path / "cancer_copy_number.tsv", sep='\t', index_col=[0], header=[1])
    cancer_copy_number = cancer_copy_number.drop(cancer_copy_number.iloc[0].name)
    cancer_discretized_copy_number = pd.read_csv(csa_x_path / "cancer_discretized_copy_number.tsv", sep='\t', index_col=[0], header=[1])
    cancer_discretized_copy_number = cancer_discretized_copy_number.drop(cancer_discretized_copy_number.iloc[0].name)
    cancer_gene_expression = pd.read_csv(csa_x_path / "cancer_gene_expression.tsv", sep='\t', index_col=[0], header=[2])
    cancer_mutation_count = pd.read_csv(csa_x_path / "cancer_mutation_count.tsv", sep='\t', index_col=[0], header=[1])
    cancer_mutation_count = cancer_mutation_count.drop(cancer_mutation_count.iloc[0].name)
    # NOT GENE SYMBOL
    cancer_RPPA = pd.read_csv(csa_x_path / "cancer_RPPA.tsv", sep='\t', index_col=[0], header=[0])
    # NOT GENE SYMBOL
    cancer_miRNA_expression = pd.read_csv(csa_x_path / "cancer_miRNA_expression.tsv", sep='\t', index_col=[0], header=[0])
    # NOT GENE SYMBOL
    cancer_DNA_methylation = pd.read_csv(csa_x_path / "cancer_DNA_methylation.tsv", sep='\t', index_col=[0], header=[0])
    cancer_DNA_methylation = cancer_DNA_methylation.drop(cancer_DNA_methylation.iloc[0].name)

    # at least the methylation has invalid NaN values, check all
    print('copy_number')
    cancer_copy_number = check_valid_NaNs(cancer_copy_number)
    print('discretized_copy_number')
    cancer_discretized_copy_number = check_valid_NaNs(cancer_discretized_copy_number)
    print('gene_expression')
    cancer_gene_expression = check_valid_NaNs(cancer_gene_expression)
    print('mutation_count')
    cancer_mutation_count = check_valid_NaNs(cancer_mutation_count)
    print('RPPA')
    cancer_RPPA = check_valid_NaNs(cancer_RPPA)
    print('miRNA_expression')
    cancer_miRNA_expression = check_valid_NaNs(cancer_miRNA_expression)
    print('DNA_methylation')
    cancer_DNA_methylation = check_valid_NaNs(cancer_DNA_methylation)

    cancer_copy_number.to_csv(output_x_path / "cancer_copy_number.tsv", sep='\t', index_label='improve_sample_id')
    cancer_discretized_copy_number.to_csv(output_x_path / "cancer_discretized_copy_number.tsv", sep='\t', index_label='improve_sample_id')
    cancer_gene_expression.to_csv(output_x_path / "cancer_gene_expression.tsv", sep='\t', index_label='improve_sample_id')
    cancer_mutation_count.to_csv(output_x_path / "cancer_mutation_count.tsv", sep='\t', index_label='improve_sample_id')
    cancer_RPPA.to_csv(output_x_path / "cancer_RPPA.tsv", sep='\t', index_label='improve_sample_id')
    cancer_miRNA_expression.to_csv(output_x_path / "cancer_miRNA_expression.tsv", sep='\t', index_label='improve_sample_id')
    cancer_DNA_methylation.to_csv(output_x_path / "cancer_DNA_methylation.tsv", sep='\t', index_label='improve_sample_id')
    del cancer_copy_number, cancer_discretized_copy_number, cancer_gene_expression, cancer_mutation_count, cancer_RPPA, cancer_miRNA_expression, cancer_DNA_methylation

    #################################### Y DATA #########################################
    # adds split ID
    # adds NCI-60 data 

    drp_response = pd.read_csv(csa_y_path / "response.tsv", sep='\t', header=[0])
    combined_single_response_agg = pd.read_csv(input_dir / "combined_single_response_agg", sep="\t")
    sample_info = pd.read_csv(input_dir / "sample_info.csv")

    nci60_response = combined_single_response_agg[combined_single_response_agg['SOURCE'] == 'NCI60']
    nci60_response['CELL'] = nci60_response['CELL'].str.replace('NCI60.', '')
    nci60_response['CELL'] = nci60_response['CELL'].str.replace(' ', '')
    nci60_response['CELL'] = nci60_response['CELL'].str.replace('-', '')
    nci60_response['CELL'] = nci60_response['CELL'].str.upper()
    sample_info = sample_info[['DepMap_ID', 'stripped_cell_line_name']]
    sample_info.columns = ['improve_sample_id', 'CELL']
    nci60_response = sample_info.merge(nci60_response)
    nci60_response = nci60_response.drop('CELL', axis=1)
    nci60_response = nci60_response.rename(columns={'SOURCE': 'source', 'DRUG': 'improve_chem_id', 'STUDY': 'study', 'AUC': 'auc', 'IC50': 'ic50', 'EC50': 'ec50', 'EC50se': 'ec50se', 'R2fit': 'r2fit', 'Einf': 'einf', 'HS': 'hs', 'AAC1': 'aac1', 'AUC1': 'auc1', 'DSS1': 'dss1'})
    drp_columns = drp_response.columns.tolist()
    nci60_response = nci60_response[drp_columns]
    new_response = pd.concat([drp_response, nci60_response], ignore_index=True)


    #################################### DRUG DATA #########################################
    drp_drugs = pd.read_csv(csa_x_path / "drug_SMILES.tsv", sep='\t', header=[0])
    # NCI60 drugs

    ChemStructures_Consistent = pd.read_csv(input_dir / "ChemStructures_Consistent.smiles", sep="\t")
    ChemStructures_Consistent['improve_chem_id'] = 'NSC.' + ChemStructures_Consistent['NSC'].astype(str)
    nci60_drugs = ChemStructures_Consistent[ChemStructures_Consistent['improve_chem_id'].isin(nci60_response['improve_chem_id'].tolist())]
    nci60_drugs = nci60_drugs.rename(columns={'SMILES': 'canSMILES'})
    nci60_drugs = nci60_drugs[['improve_chem_id', 'canSMILES']].reset_index(drop=True)

    # check that these are canonical
    good, bad = canonicalize_smiles(nci60_drugs, id_col_name='improve_chem_id', smiles_col_name='canSMILES')

    new_drugs = pd.concat([drp_drugs, good], ignore_index=True)
    print(new_drugs)
    # save smiles
    new_drugs.to_csv(output_x_path / "drug_SMILES.tsv", sep='\t', index=False)
    bad.to_csv(output_x_path / "bad_SMILES.tsv", sep='\t', index=False)

    # save fingerprints
    drug_ecfp4_nbits512 = generate_fingerprints(new_drugs, radius=2, nbits=512, smiles_col_name='canSMILES')
    drug_ecfp4_nbits512.to_csv(output_x_path / "drug_ecfp4_nbits512.tsv", sep='\t', index=False)

    # save descriptors
    mordred, _ = generate_mordred(new_drugs, smiles_col_name='canSMILES')
    mordred.to_csv(output_x_path / "drug_mordred.tsv", sep='\t', index=False)

    new_response = new_response[new_response['improve_chem_id'].isin(new_drugs['improve_chem_id'].tolist())]
    new_response = new_response.reset_index().rename(columns={'index': 'split_id'})
    new_response.to_csv(output_y_path / "response.tsv", sep='\t')

    #################################### SPLITS #########################################
    response_only_nci60 = new_response[new_response['source'] == 'NCI60']
    generate_mixed_splits(response_only_nci60, study_col='source', output_dir=str(output_splits_path))
    generate_blind_splits(new_response, study_col='source', blind_col='improve_sample_id', blind_name='cell', output_dir=str(output_splits_path))
    generate_blind_splits(new_response, study_col='source', blind_col='improve_chem_id', blind_name='drug', output_dir=str(output_splits_path))

def main(args):
    args = parse_args(args)
    run(args)

if __name__ == '__main__':
    main(sys.argv[1:])