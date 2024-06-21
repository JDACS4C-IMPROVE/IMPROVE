import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import sys
import os

from improvelib.Benchmarks.Base import DataStager
from improvelib.Benchmarks.DrugResponsePrediction import *

# [Req] IMPROVE/CANDLE imports

def test_benchmark(benchmark):
    # Configuring benchmark for getting a particular data split
    benchmark.set_dataset(SingleDRPDataset.CCLE)
    benchmark.set_split_id(0)
    benchmark.set_split_type(SplitType.TEST)
    benchmark.set_drp_metric(DRPMetric.AUC)

    # Getting dataframes from IMPROVE project for one split
    response = benchmark.get_dataframe(SingleDRPDataFrame.RESPONSE)
    gene_expression = benchmark.get_dataframe(
        SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION)
    drug_smiles = benchmark.get_dataframe(SingleDRPDataFrame.DRUG_SMILES)

    # Get all data in SMILES dataframe
    gene_expression_full = benchmark.get_full_dataframe(
        SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION)


def test_data_staging(benchmark, output_dir):
    data_stager = DataStager()
    data_stager.set_benchmark(benchmark)
    data_stager.set_output_dir(output_dir)
    staged_files_dict = data_stager.stage_all_experiments([SingleDRPDataset.CCLE],
                                                          [SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION,
                                                           SingleDRPDataFrame.DRUG_SMILES,
                                                           SingleDRPDataFrame.RESPONSE],
                                                          DRPMetric.AUC)
    print(staged_files_dict)


if __name__ == '__main__':
    benchmark_dir = os.environ['IMPROVE_BENCHMARK_DIR'] #os.path.join(os.environ['IMPROVE_DATA_DIR'], 'raw_data')
    staging_dir = './data_staging'

    benchmark = SingleDRPBenchmark()
    benchmark.set_benchmark_dir(benchmark_dir)

    # test_benchmark(benchmark)
    test_data_staging(benchmark, staging_dir)
