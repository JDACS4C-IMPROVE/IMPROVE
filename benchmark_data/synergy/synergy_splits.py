import pandas as pd
import numpy as np
from benchmark_data.splits.splits_generator import generate_mixed_splits, generate_blind_splits

df = pd.read_csv("./synergy.tsv", sep='\t')


generate_mixed_splits(df)
generate_blind_splits(df, blind_col='DepMapID', blind_name='cell')