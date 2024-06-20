import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import sys
import os

sys.path.append('/Users/onarykov/git/improve-lib/IMPROVE')
#########
#########
#########


# [Req] IMPROVE/CANDLE imports

def test_benchmark(benchmark):
    # Configuring benchmark for getting a particular data split
    benchmark.set_dataset(drp.SingleDRPDataset.CCLE)
    benchmark.set_split_id(0)
    benchmark.set_split_type(drp.SplitType.TEST)
    benchmark.set_drp_metric(drp.DRPMetric.AUC)

    # Getting dataframes from IMPROVE project for one split
    response = benchmark.get_dataframe(drp.SingleDRPDataFrame.RESPONSE)
    gene_expression = benchmark.get_dataframe(
        drp.SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION)
    drug_smiles = benchmark.get_dataframe(drp.SingleDRPDataFrame.DRUG_SMILES)

    # Get all data in SMILES dataframe
    gene_expression_full = benchmark.get_full_dataframe(
        drp.SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION)


def test_data_staging(benchmark, output_dir):
    data_stager = drp.SingleDRPDataStager()
    data_stager.set_benchmark(benchmark)
    data_stager.set_output_dir(output_dir)
    staged_files_dict = data_stager.stage_all_experiments([drp.SingleDRPDataset.CCLE],
                                                          [drp.SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION,
                                                           drp.SingleDRPDataFrame.DRUG_SMILES,
                                                           drp.SingleDRPDataFrame.RESPONSE],
                                                          drp.DRPMetric.AUC)
    print(staged_files_dict)


if __name__ == '__main__':
    from improvelib import framework as frm
    from improvelib import drug_resp_pred as drp

    benchmark_dir = os.path.join(os.environ['IMPROVE_DATA_DIR'], 'raw_data')
    staging_dir = './data_staging'

    benchmark = drp.SingleDRPBenchmark()
    benchmark.set_benchmark_dir(benchmark_dir)

    # test_benchmark(benchmark)
    test_data_staging(benchmark, staging_dir)
