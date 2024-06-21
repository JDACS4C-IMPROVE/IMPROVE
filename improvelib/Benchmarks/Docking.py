from improvelib.Benchmarks.Base import Benchmark, Stage, ParameterConverter
from improvelib.Benchmarks.benchmark_utils import StringEnum
from improvelib.data_loader import DataInputOutput
import pandas as pd
import os


class DockingStage(StringEnum):
    """
    Enum for specifying the docking stage.
    """
    TRAIN = 'tr'
    VALIDATION = 'vl'
    TEST = 'te'


class DockingDataFrame(StringEnum):
    """
    Enum for specifying docking dataframes.
    """

    DOCKING_SCORES = 'docking_scores'
    PROTEIN_DESCRIPTORS = 'protein_descriptors'
    DRUG_ECFP2 = 'drug_ecfp2'


class DockingDataset(StringEnum):
    """
    Enum for specifying datasets in single drug response prediction benchmarks.
    """
    ZINC_DB = 'DIR.ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col'


class DockingMetric(StringEnum):
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
    DRUG_ID_COL = 'TITLE'
    PROTEIN_ID_COL = 'TITLE'

    def __init__(self):
        """
        Initializes the SingleDRPBenchmark instance with default values.
        """
        self._initialize()
        self._SPLITS_NUM = 19
        self._splits_ids = list(range(self._SPLITS_NUM))
        self._loaded_dfs = {}
        self._splits_dir = 'ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.splits'
        self._dataset2file_map = {
            DockingDataFrame.DOCKING_SCORES: "docks.df.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.csv",
            DockingDataFrame.PROTEIN_DESCRIPTORS: "ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet",
            DockingDataFrame.DRUG_ECFP2: "ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.ecfp2.parquet"
        }
        loaded_dfs = {}

    def get_datasets(self):
        return DockingDataset

    def get_dataframes(self):
        return DockingDataFrame

    def get_stages(self):
        return DockingStage

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

    def get_dataframe(self, dataframe):
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
                             f'{str(self._stage)}', 'id.csv'))
        return filename

    def _load_raw_dataframe(self, dataframe):
        if dataframe not in self._dataset2file_map:
            raise Exception(
                f"Dataframe {dataframe} is not mapped to the file!")

        data_loader = DataInputOutput()
        file_name = self._dataset2file_map[dataframe]
        file_type = file_name.split('.')[-1]
        # HACK: fix for the uppercase for csv type
        if file_type == 'csv':
            file_type = file_type.upper()
        #

        file_path = os.path.join(
            self._benchmark_dir,
            str(self._dataset),
            file_name)
        if file_path is None:
            raise Exception(f'Location of {self._dataset} is not set!')
        data = data_loader.load_data(file_path, file_type)
        return data

    def _load_split_ids(self):
        file_name = self._construct_splits_file_name()
        file_path = os.path.join(
            self._benchmark_dir, str(self._dataset), self._splits_dir, file_name)
        data_loader = DataInputOutput()

        data = data_loader.load_data(file_path, 'CSV').values.flatten()
        return data

    def _load_docking_scores(self):
        df = self._load_raw_dataframe(DockingDataFrame.DOCKING_SCORES)
        split_ids = self._load_split_ids()
        df = df.iloc[split_ids]
        cols_to_drop = [col for col in df.columns if col not in [
            self.DRUG_ID_COL, self.PROTEIN_ID_COL, str(self._metric)]]
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    def _load_dataframe(self, dataframe):
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
            if dataframe not in self._dataset2file_map:
                raise Exception(
                    f'Dataframe name {dataframe} is not mapped to the file!')
            if 'scores' in str(dataframe):
                return self._load_docking_scores()
            df = self._load_raw_dataframe(dataframe)
            self._loaded_dfs[dataframe] = df

        docking_scores = self._load_docking_scores()
        key_col_name = None
        if 'protein' in dataframe.value:
            key_col_name = self.PROTEIN_ID_COL
        elif 'drug' in dataframe.value:
            key_col_name = self.DRUG_ID_COL

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
    proteins = benchmark.get_dataframe(dataframes.PROTEIN_DESCRIPTORS)
    breakpoint()
