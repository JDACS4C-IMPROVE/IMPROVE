
conda activate benchmark_data
export PYTHONPATH=/PATH/TO/IMPROVE/

python generate_random_data.py random_SMILES --input_file /PATH/TO/x_data/drug_SMILES.tsv --output_file /PATH/TO/OUTPUT/drug_SMILES_drugX_1.tsv
python generate_random_data.py random_SMILES --input_file /PATH/TO/x_data/drug_SMILES.tsv --output_file /PATH/TO/OUTPUT/drug_SMILES_drugX_2.tsv --seed 43
python generate_random_data.py random_SMILES --input_file /PATH/TO/x_data/drug_SMILES.tsv --output_file /PATH/TO/OUTPUT/drug_SMILES_drugX_3.tsv --seed 44


python generate_random_data.py shuffle --input_file /PATH/TO/x_data/cancer_gene_expression.tsv --output_file /PATH/TO/OUTPUT/cancer_gene_expression_shuffle_full_1.tsv 
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/cancer_gene_expression.tsv --output_file /PATH/TO/OUTPUT/cancer_gene_expression_shuffle_full_2.tsv --seed 43
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/cancer_gene_expression.tsv --output_file /PATH/TO/OUTPUT/cancer_gene_expression_shuffle_full_3.tsv --seed 44

python generate_random_data.py shuffle --input_file /PATH/TO/x_data/drug_mordred.tsv --output_file /PATH/TO/OUTPUT/drug_mordred_shuffle_full_1.tsv 
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/drug_mordred.tsv --output_file /PATH/TO/OUTPUT/drug_mordred_shuffle_full_2.tsv --seed 43
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/drug_mordred.tsv --output_file /PATH/TO/OUTPUT/drug_mordred_shuffle_full_3.tsv --seed 44









