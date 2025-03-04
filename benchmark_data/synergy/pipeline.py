import sys
import pandas as pd
import numpy as np
import argparse
from chem_utils import canonicalize_smiles, generate_fingerprints, generate_mordred, generate_infomax

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

def run(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    ### load drugcomb synergy data
    drugcomb = pd.read_csv(input_dir + "drugcomb_summary_v_1_5.csv")

    # combine small studies and drop studies that aren't cancer
    NCATS_variants = ["NCATS_ES(FAKI/AURKI)", "NCATS_ES(NAMPT+PARP)", "NCATS_HL", "NCATS_2D_3D"]
    NCATS_variants2 = ["NCATS_DIPG", "NCATS_MDR_CS", "NCATS_ATL"]
    small_variants = ["YOHE", "WILSON", "SCHMIDT", "PHELAN", "FALLAHI-SICHANI", "MILLER"]
    fried_variants = ["FRIEDMAN", "FRIEDMAN2"]
    not_cancer = ["MOTT", "DYALL", "NCATS_SARS-COV-2DPI", "BOBROWSKI"]

    drugcomb['study_name'] = drugcomb['study_name'].apply(lambda x: 'NCATS-1' if x in NCATS_variants else x)
    drugcomb['study_name'] = drugcomb['study_name'].apply(lambda x: 'NCATS-2' if x in NCATS_variants2 else x)
    drugcomb['study_name'] = drugcomb['study_name'].apply(lambda x: 'SMALL' if x in small_variants else x)
    drugcomb['study_name'] = drugcomb['study_name'].apply(lambda x: 'FRIEDMAN' if x in fried_variants else x)
    drugcomb = drugcomb.dropna(subset=['drug_col'])
    drugcomb = drugcomb[~drugcomb['study_name'].isin(not_cancer)]

    # get stripped name to match to IDs
    drugcomb['stripped_cell_line_name'] = drugcomb['cell_line_name'].str.replace('-', '')
    drugcomb['stripped_cell_line_name'] = drugcomb['stripped_cell_line_name'].str.replace(' ', '')
    drugcomb['stripped_cell_line_name'] = drugcomb['stripped_cell_line_name'].str.upper()

    # get DepMap IDs and hand-curated DepMap IDs
    depmap_ids = pd.read_csv(input_dir + "sample_info.csv")
    depmap_ids = depmap_ids[['DepMap_ID', 'stripped_cell_line_name']]
    depmap_curated = pd.read_csv(input_dir + "drugcomb_BIG_notindepmap_curated.csv")
    depmap_curated = depmap_curated[depmap_curated['DepMap_ID'].notna()]
    depmap_curated['stripped_cell_line_name'] = depmap_curated['cell_line_name'].str.replace('-', '')
    depmap_curated['stripped_cell_line_name'] = depmap_curated['stripped_cell_line_name'].str.replace(' ', '')
    depmap_curated['stripped_cell_line_name'] = depmap_curated['stripped_cell_line_name'].str.upper()
    depmap_curated = depmap_curated[['DepMap_ID', 'stripped_cell_line_name']]
    depmap_ids = pd.concat([depmap_ids, depmap_curated])
    # join DepMap IDs to response data
    drugcomb_depmap = drugcomb.merge(depmap_ids, on='stripped_cell_line_name', how='left')
    drugcomb_depmap = drugcomb_depmap[drugcomb_depmap['DepMap_ID'].notna()]

    # cell lines and drugs in y_data
    cell_lines = drugcomb_depmap['DepMap_ID'].unique()
    drugs = pd.concat([drugcomb['drug_row'], drugcomb['drug_col']], axis=0).unique()

    #################################### MUTATION #########################################
    # DepMap data 22Q2 has the same cell lines as 23Q4 so we use that for consistency with mutation data
    MUT_22Q2 = pd.read_csv(input_dir + "CCLE_mutations.csv")
    MUT_22Q2['names'] = MUT_22Q2['Entrez_Gene_Id'].astype(str)

    ### mutation data for deleterious only, binarized
    MUT_22Q2['isDeleterious'] = MUT_22Q2['isDeleterious'].astype(str)
    MUT_22Q2_del = MUT_22Q2[MUT_22Q2['isDeleterious'] == 'True']
    MUT_22Q2_del = MUT_22Q2_del[['DepMap_ID', 'names']]
    MUT_22Q2_del['del_mut'] = 1
    MUT_22Q2_del = MUT_22Q2_del.drop_duplicates()
    MUT_22Q2_del_wide = MUT_22Q2_del.pivot(index='DepMap_ID', columns='names', values='del_mut')
    MUT_22Q2_del_wide = MUT_22Q2_del_wide.fillna(0)
    MUT_22Q2_del_wide = MUT_22Q2_del_wide.rename_axis(None, axis=1).reset_index()
    MUT_22Q2_del_wide = MUT_22Q2_del_wide[MUT_22Q2_del_wide['DepMap_ID'].isin(cell_lines)]
    MUT_22Q2_del_wide.to_csv(output_dir + "cell_mutation_delet.tsv", sep='\t', index=False)

    ### mutation data for all non-silent mutations, binarized
    MUT_22Q2_notSilent = MUT_22Q2[MUT_22Q2['Variant_annotation'] != 'silent']
    MUT_22Q2_notSilent = MUT_22Q2_notSilent[['DepMap_ID', 'names']]
    MUT_22Q2_notSilent['mut'] = 1
    MUT_22Q2_notSilent = MUT_22Q2_notSilent.drop_duplicates()
    MUT_22Q2_notSilent_wide = MUT_22Q2_notSilent.pivot(index='DepMap_ID', columns='names', values='mut')
    MUT_22Q2_notSilent_wide = MUT_22Q2_notSilent_wide.fillna(0)
    MUT_22Q2_notSilent_wide = MUT_22Q2_notSilent_wide.rename_axis(None, axis=1).reset_index()
    MUT_22Q2_notSilent_wide = MUT_22Q2_notSilent_wide[MUT_22Q2_notSilent_wide['DepMap_ID'].isin(cell_lines)]
    MUT_22Q2_notSilent_wide.to_csv(output_dir + "cell_mutation_nonsynon.tsv", sep='\t', index=False)

    #################################### COPY NUMBER #########################################
    CNV_22Q2 = pd.read_csv(input_dir + "CCLE_gene_cn.csv")
    CNV_22Q2.rename(columns={CNV_22Q2.columns[0]: "DepMapID" }, inplace = True)
    CNV_22Q2.set_index(CNV_22Q2.columns[0], inplace=True)
    cols = CNV_22Q2.columns.to_list()
    new_cols = []
    for c in cols:
        cc = [c.split('(')[1].split(')')[0]]
        new_cols = new_cols + cc
    CNV_22Q2.columns = new_cols
    CNV_22Q2 = CNV_22Q2.reset_index()

    ### continuous
    CNV_22Q2_cont = CNV_22Q2[CNV_22Q2['DepMapID'].isin(cell_lines)]
    CNV_22Q2_cont.to_csv(output_dir + "cell_cnv_continuous.tsv", sep='\t', index=False)

    ### discretized
    ### deep del < 0.5210507 < het loss < 0.7311832 < diploid < 1.214125 < gain < 1.422233 < amp
    ### from Priyanka
    ### reference: Mina, Marco, et al. "Discovering functional evolutionary dependencies in human cancers." Nature genetics 52.11 (2020): 1198-1207.
    CNV_22Q2_bins = CNV_22Q2_cont.set_index('DepMapID')
    bins = [-np.inf, 0.5210507, 0.7311832, 1.214125, 1.422233, np.inf]
    labels = [-2, -1, 0, 1, 2]
    CNV_22Q2_bins = CNV_22Q2_bins.apply(lambda x: pd.cut(x, bins=bins, labels=labels)).reset_index()
    CNV_22Q2_bins.to_csv(output_dir + "cell_cnv_discretized.tsv", sep='\t', index=False)


    #################################### GENE EXPRESSION #########################################
    GE_22Q2 = pd.read_csv(input_dir + "CCLE_expression.csv")
    GE_22Q2.rename(columns={GE_22Q2.columns[0]: "DepMapID" }, inplace = True)
    GE_22Q2.set_index(GE_22Q2.columns[0], inplace=True)
    cols = GE_22Q2.columns.to_list()
    new_cols = []
    for c in cols:
        cc = [c.split('(')[1].split(')')[0]]
        new_cols = new_cols + cc
    GE_22Q2.columns = new_cols
    GE_22Q2 = GE_22Q2.reset_index()
    GE_22Q2_all = GE_22Q2[GE_22Q2['DepMapID'].isin(cell_lines)]
    GE_22Q2_all.to_csv(output_dir + "cell_transcriptomics.tsv", sep='\t', index=False)

    #################################### DRUGS - SMILES ####################################

    ### drugcomb drug info
    # drug info from: 
    # converted from a list of dictionaries to df to csv in python
    drugcomb_drugs = pd.read_csv(input_dir + "drugcomb_drugs_df.csv")
    drugcomb_drugs = drugcomb_drugs[drugcomb_drugs['dname'].isin(drugs)]

    # these are all pairs with the same smiles
    # kept based on:
    # 1) one had a chembl_id
    # 2) one had more of drugbank/kegg/target
    # 3) the lower id number as tie breaker
    samesmiles_todrop = [3375, 3576, 389, 8370, 8334, 2665, 2387, 2831, 4213, 982, 2614, 2837, 2838, 8391, 2812,
                        2826, 1641, 2741, 4276, 3141, 3795, 899, 2361, 2902, 2861, 2908, 4289, 
                        243, 433, 3003, 2807, 1344, 2386, 2625, 3631, 2631, 2711, 2588, 8336, 4124,
                        2827, 4348, 2818, 147, 3021, 117, 3506, 4225, 4630, 119]
    # these are pairs where one (or more) of the smiles is iso or null
    isonullsmiles_todrop = [6147, 8166, 7133, 
                            58, 2105, 2819, 7848, 6285, 7711, 5671, 7479, 6700, 8157, 
                            7586, 4329, 7451, 8009, 8140,
                            5949, 7334, 384, 7114, 7574, 8068]
    # the remaining smiles were checked and compared to structure (wiki, selleckchem, etc)
    remaining_todrop = [1655, 1906, 2144, 6070, 5190, 7557, 177, 1801, 3196, 7446, 5614, 5354, 1425, 5451, 2204, 4196, 6153, 7888, 7582, 3908, 2266, 1755, 2850, 4138, 7950, 1205, 1318, 1410, 1434, 5389, 2996, 1542]
    drugcomb_drugs = drugcomb_drugs[~drugcomb_drugs['id'].isin(samesmiles_todrop)]
    drugcomb_drugs = drugcomb_drugs[~drugcomb_drugs['id'].isin(isonullsmiles_todrop)]
    drugcomb_drugs = drugcomb_drugs[~drugcomb_drugs['id'].isin(remaining_todrop)]
    #duped_now = drugcomb_drugs[drugcomb_drugs.duplicated(subset=['dname'], keep=False)]
    #print(len(duped_now))
    #df_subset = drugcomb_drugs[drugcomb_drugs['smiles'].str.contains(';', na=False)]
    multiplesmiles_curated = pd.read_csv(input_dir + "checkedsmi_curated.csv")
    drugcomb_drugs_final = drugcomb_drugs
    for index, row in multiplesmiles_curated.iterrows():
        drugcomb_drugs_final.loc[drugcomb_drugs_final['id'] == row['id'], 'smiles'] = row['correct_smile']

    drugcomb_drugs_final['id'] = 'drug_' + drugcomb_drugs_final['id'].astype(str)
    smiles = drugcomb_drugs_final[['id', 'smiles']]
    smiles.rename(columns={'id': 'DrugID'}, inplace=True)
    smiles.to_csv(output_dir + "drug_smiles.tsv", sep='\t', index=False)

    smi_df = pd.read_csv('drug_smiles.tsv', sep='\t')

    #################################### DRUGS - OTHER ####################################
    smiles = smiles.reset_index(drop=True)
    good, bad = canonicalize_smiles(smiles)
    bad.to_csv(output_dir + "drug_smiles_bad.tsv", sep='\t', index=False)
    good.to_csv(output_dir + "drug_smiles_canonical.tsv", sep='\t', index=False)

    drug_ecfp2_nbits256 = generate_fingerprints(good, radius=1, nbits=256)
    drug_ecfp4_nbits256 = generate_fingerprints(good, radius=2, nbits=256)
    drug_ecfp6_nbits256 = generate_fingerprints(good, radius=3, nbits=256)
    drug_ecfp2_nbits1024 = generate_fingerprints(good, radius=1, nbits=1024)
    drug_ecfp4_nbits1024 = generate_fingerprints(good, radius=2, nbits=1024)
    drug_ecfp6_nbits1024 = generate_fingerprints(good, radius=3, nbits=1024)

    drug_ecfp2_nbits256.to_csv(output_dir + "drug_ecfp2_nbits256.tsv", sep='\t', index=False)
    drug_ecfp4_nbits256.to_csv(output_dir + "drug_ecfp4_nbits256.tsv", sep='\t', index=False)
    drug_ecfp6_nbits256.to_csv(output_dir + "drug_ecfp6_nbits256.tsv", sep='\t', index=False)
    drug_ecfp2_nbits1024.to_csv(output_dir + "drug_ecfp2_nbits1024.tsv", sep='\t', index=False)
    drug_ecfp4_nbits1024.to_csv(output_dir + "drug_ecfp4_nbits1024.tsv", sep='\t', index=False)
    drug_ecfp6_nbits1024.to_csv(output_dir + "drug_ecfp6_nbits1024.tsv", sep='\t', index=False)

    mordred, _ = generate_mordred(good)
    mordred.to_csv(output_dir + "drug_mordred.tsv", sep='\t', index=False)

    infomax = generate_infomax(good)
    infomax.to_csv(output_dir + "drug_infomax.tsv", sep='\t', index=False)

    #################################### SYNERGY ####################################
    drug_ids = drugcomb_drugs_final[['id', 'dname']]
    synergy = drugcomb_depmap.merge(drug_ids, left_on='drug_row', right_on='dname', how='left')
    synergy = synergy.merge(drug_ids, left_on='drug_col', right_on='dname', how='left')
    columns_to_keep = ['DepMap_ID', 'id_x', 'id_y', 'study_name', 'synergy_loewe', 'synergy_bliss', 'synergy_zip', 'synergy_hsa', 'S_mean', 'css_ri']
    synergy = synergy[columns_to_keep]
    synergy.rename(columns={'DepMap_ID': 'DepMapID', 'id_x': 'DrugID_row', 'id_y': 'DrugID_col', 'study_name': 'study', 'synergy_loewe': 'loewe', 'synergy_bliss': 'bliss', 'synergy_zip': 'zip', 'synergy_hsa': 'hsa', 'S_mean': 'smean', 'css_ri': 'css'}, inplace=True)
    synergy.replace('\\N', np.nan, inplace=True)
    synergy.to_csv(output_dir + "synergy.tsv", sep='\t', index=False)

def main(args):
    args = parse_args(args)
    run(args)

if __name__ == '__main__':
    main(sys.argv[1:])