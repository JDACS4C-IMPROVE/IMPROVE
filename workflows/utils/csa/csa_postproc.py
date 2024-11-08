"""Cross-study analysis (CSA) post-processing script.

This script performs post-processing analysis on cross-study results, including
runtime performance analysis and prediction performance analysis.

Example:
    python csa_postproc.py --res_dir res.csa --model_name GraphDRP --y_col_name auc
    
Command-line arguments:
    --res_dir: Directory containing the CSA results.
    --model_name: Name of the model (e.g., GraphDRP, DeepCDR).
    --y_col_name: Name of the target column. Defaults to 'auc'.
    --outdir: Directory to save post-processing results. Optional.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from csa_utils import (
    runtime_analysis,
    csa_postprocess
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments, including:
            --res_dir: Directory containing the CSA results.
            --model_name: Name of the model (e.g., GraphDRP, DeepCDR).
            --y_col_name: Name of the target column. Defaults to 'auc'.
            --outdir: Directory to save post-processing results. Optional.
    """
    parser = argparse.ArgumentParser(description='Cross-study analysis post-processing.')
    parser.add_argument('--res_dir',
                        type=str,
                        required=True,
                        help='Directory containing the CSA results.')
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='Name of the model (e.g., GraphDRP, DeepCDR).')
    parser.add_argument('--y_col_name',
                        type=str,
                        default='auc',
                        required=False,
                        help='Name of the target column. Defaults to \'auc\'.')
    parser.add_argument('--outdir',
                        type=str,
                        default=None,
                        required=False,
                        help='Directory to save post-processing results. Optional.')
    return parser.parse_args()


def analyze_runtime(res_dir_path: Path, stage_mapping: dict) -> pd.DataFrame:
    """Analyze runtime performance for different stages.
    
    Args:
        res_dir_path: Path to the CSA result directory.
        stage_mapping: Mapping of directory names to stage names.
        
    Returns:
        pd.DataFrame: Concatenated DataFrame of runtime analysis results.
        None: If no runtime data is found.
    """
    times = []
    for stage_dir_name, stage_name in stage_mapping.items():
        # Analyze runtime for each stage
        df = runtime_analysis(res_dir_path,
                            stage_dir_name,
                            model_name,
                            decimal_places=4)
        if df is not None:
            df['stage'] = stage_name
            times.append(df)
    
    if len(times) > 0:
        # Concatenate results from all stages
        return pd.concat(times, axis=0)
    return None


def main():
    """Execute CSA post-processing analysis pipeline."""
    args = parse_arguments()
    
    # Setup paths
    res_dir_path = Path(args.res_dir).resolve()
    if args.outdir is None:
        # Default output directory if not specified
        outdir = res_dir_path.parent / f'postproc.csa.{args.model_name}.{res_dir_path.name}'
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Runtime analysis
    stage_mapping = {
        'ml_data': 'preprocess',
        'models': 'train',
        'infer': 'infer'
    }
    
    times_df = analyze_runtime(res_dir_path, stage_mapping)
    if times_df is not None:
        # Save runtime analysis results
        times_df.to_csv(outdir / "runtimes.csv", index=False)
    
    # Prediction performance analysis
    scores = csa_postprocess(res_dir_path,
                           args.model_name,
                           args.y_col_name,
                           decimal_places=4,
                           outdir=outdir)
    
    print('\nFinished cross-study post-processing.')


if __name__ == "__main__":
    main()
