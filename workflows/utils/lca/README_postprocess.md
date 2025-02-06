## Postprocessing of LCA

## Usage

**Python script**: `lca_postprocess.py`
This script generates postprocessing of learning curve analysis data, including run-time analysis and scores.


To generate run-time analysis:
```bash
python lca_postprocess.py runtimes <arguments>
```
This will output a table `runtimes.csv` in the specified `output_dir`.

To generate aggregate scores:
```bash
python lca_postprocess.py lca_scores <arguments>
```
This will output a table `all_scores.csv` in the specified `output_dir`.

To generate learning curve plot:
```bash
python lca_postprocess.py plot_learning_curve <arguments>
```
This will use `all_scores.csv` the specified `output_dir` and output a plot `fig.png` in the specified `output_dir`.

To run all analyses:
```bash
python lca_postprocess.py whole_analysis <arguments>
```
This will run the run-time analysis, aggregate scores, and plot the learning curve.

**Arguments**:
* `--input_dir`: Path to the LCA results (default: `'./'`).
* `--output_dir`: Path to the directory where the postprocessing will be saved (default: `'./'`).
* `--y_col_name`: The y_col_name in `test_y_data_predicted.csv` (default: `'auc'`).
* `--metric_type`: Metric type to use (default: `'regression'`).
* `--model_name`: Name of the model, if you would like it saved in the data / title of the plot (default: `None`).
* `--dataset`: Name of the dataset, if you would like it saved in the data / title of the plot (default: `None`).



