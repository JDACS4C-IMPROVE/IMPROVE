from improvelib.benchmarks.base import Benchmark, Stage, ParameterConverter, DataFrameDescription, DatasetDescription
from improvelib.data_loader import DataInputOutput
import pandas as pd
import os
from enum import Enum


DRUG_KEY_COL = 'TITLE'
DOCKING_KEY_COL = 'TITLE'


class DockingDataFrame(Enum):
    """
    Enum for specifying docking dataframes.
    """

    DOCKING_SCORES = DataFrameDescription(name='docking_scores',
                                          file='docks.df.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.csv',
                                          group='docking',
                                          key_column=DOCKING_KEY_COL,
                                          description='Docking scores')
    DRUG_DESCRIPTORS = DataFrameDescription(name='drug_descriptors',
                                            file='ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet',
                                            group='drug',
                                            key_column=DRUG_KEY_COL,
                                            description='Computational drug descriptors')

    DRUG_ECFP2 = DataFrameDescription(name='drug_ecfp2',
                                      file='ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.ecfp2.parquet',
                                      group='drug',
                                      key_column=DRUG_KEY_COL,
                                      description='Computational drug descriptors')


class DockingDataset(Enum):
    """
    Enum for specifying datasets in single drug response prediction benchmarks.
    """
    ZINC_DB = DatasetDescription(name='DIR.ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col',
                                 description='Computational docking (XXX tool) for YYY compounds over Zinc DB proteins')


class DockingMetric(Enum):
    """
    Enum for specifying the drug response prediction metrics.
    """
    REGRESSION = 'reg'
    CLASSIFICATION = 'cls'


class DockingParameterConverter(ParameterConverter):
    def _add_cli_params_to_enum_convertion(self):
        param_convertion_dict = {
            'zinc_db': DockingDataset.ZINC_DB
        }
        self._convertion_dict.update(param_convertion_dict)

    def __init__(self):
        super.__init__(
            self, [DockingDataFrame, DockingDataset, Stage, DockingMetric])
        self._add_cli_params_to_enum_convertion()


class DockingBenchmark(Benchmark):

    def _adjust_stage_name(self, stage: Stage) -> str:
        if stage == Stage.TRAIN:
            return 'tr'
        if stage == Stage.VALIDATION:
            return 'vl'
        if stage == Stage.TEST:
            return 'tr'

    def __init__(self):
        """
        Initializes the SingleDRPBenchmark instance with default values.
        """
        self._initialize()
        self._SPLITS_NUM = 19
        self._splits_ids = list(range(self._SPLITS_NUM))
        self._loaded_dfs = {}
        self._splits_dir = 'ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.splits'

    def get_datasets(self):
        return DockingDataset

    def get_dataframes(self):
        return DockingDataFrame

    def get_stages(self):
        return Stage

    def get_metrics(self):
        return DockingMetric

    # Getting data from benchmark
    def get_splits_ids(self):
        """
        Returns the number of data splits.

        Returns:
            int: The number of data splits.
        """
        return self._splits_ids

    def get_dataframe(self, dataframe: DockingDataFrame):
        """
        Retrieves a dataframe based on the specified parameters.

        Args:
            dataframe (SingleDRPDataFrame): The type of dataframe to retrieve.

        Returns:
            pd.DataFrame: The requested dataframe.
        """
        return self._load_dataframe(dataframe)

    def _construct_splits_file_name(self):
        """
        Constructs the file name for the data splits based on the current dataset and split settings.

        Returns:
            str: The constructed file name for the splits.
        """
        filename = '_'.join(('1fold', f's{str(self._split_id)}',
                             f'{self._adjust_stage_name(self._stage)}', 'id.csv'))
        return filename

    def _load_raw_dataframe(self, dataframe: DockingDataFrame):

        data_loader = DataInputOutput()
        file_name = dataframe.value.file
        file_type = file_name.split('.')[-1]
        # HACK: fix for the uppercase for csv type
        if file_type == 'csv':
            file_type = file_type.upper()
        #

        file_path = os.path.join(
            self._benchmark_dir,
            self._dataset.value.name,
            file_name)
        if file_path is None:
            raise Exception(f'Location of {self._dataset} is not set!')
        data = data_loader.load_data(file_path, file_type)
        return data

    def _load_split_ids(self):
        file_name = self._construct_splits_file_name()
        file_path = os.path.join(
            self._benchmark_dir, self._dataset.value.name, self._splits_dir, file_name)
        data_loader = DataInputOutput()

        data = data_loader.load_data(file_path, 'CSV').values.flatten()
        return data

    def _load_docking_scores(self):
        df = self._load_raw_dataframe(DockingDataFrame.DOCKING_SCORES)
        split_ids = self._load_split_ids()
        df = df.iloc[split_ids]
        cols_to_drop = [col for col in df.columns if col not in [
            DRUG_KEY_COL, str(self._metric)]]
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    def _load_dataframe(self, dataframe: DockingDataFrame):
        """
        Loads a dataframe based on the specified dataframe type, ensuring it is initialized and filtered by relevant IDs.

        Args:
            dataframe (DockingDataFrame): The type of dataframe to load.

        Returns:
            pd.DataFrame: The loaded and filtered dataframe.
        """
        self._check_initialization()
        df = None
        if dataframe in self._loaded_dfs:
            df = self._loaded_dfs[dataframe]
        else:
            if dataframe.value.group == 'scores':
                return self._load_docking_scores()
            df = self._load_raw_dataframe(dataframe)
            self._loaded_dfs[dataframe] = df

        docking_scores = self._load_docking_scores()
        key_col_name = None
        key_col_name = DRUG_KEY_COL

        ids = docking_scores[key_col_name].unique()
        self._loaded_dfs[dataframe] = df
        index_name = df.index.name
        index_name = 'index' if index_name is None else index_name
        df_split = df.reset_index(drop=False)
        df_split.set_index(key_col_name, drop=False, inplace=True)
        df_split = df_split.loc[ids]
        df_split.set_index(index_name, inplace=True, drop=True)
        return df_split


if __name__ == '__main__':
    # Path to the docking benchmark
    benchmark_dir = '/Users/onarykov/git/improve-lib/docking_benchmark'
    benchmark = DockingBenchmark()
    # Get all available benchmark options
    splits_ids = benchmark.get_splits_ids()
    datasets = benchmark.get_datasets()
    dataframes = benchmark.get_dataframes()
    stages = benchmark.get_stages()
    metrics = benchmark.get_metrics()

    benchmark.set_benchmark_dir(benchmark_dir)
    benchmark.set_dataset(datasets.ZINC_DB)
    benchmark.set_split_id(splits_ids[0])
    benchmark.set_stage(stages.TEST)
    benchmark.set_metric(metrics.REGRESSION)
    scores = benchmark.get_dataframe(dataframes.DOCKING_SCORES)
    ecfp2 = benchmark.get_dataframe(dataframes.DRUG_ECFP2)
    descriptors = benchmark.get_dataframe(dataframes.DRUG_DESCRIPTORS)
    breakpoint()
