import os
import sys
import pandas as pd
from enum import Enum
from abc import ABC, ABCMeta, abstractmethod

import improvelib.utils as frm
from typing import Any


class Params:
    pass


class Base():
    """Class to handle configuration files for Preprocessing."""


class DescriptionBase():
    def _check_initialization(self):
        if self.name is None:
            raise Exception("Name in the description cannot be None!")
        if self.description is None:
            raise Exception("Dataframe description cannot be None!")

    def __str__(self) -> str:
        return str(self.name)

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class DatasetDescription(DescriptionBase):

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self._check_initialization()


class DataFrameDescription(DescriptionBase):

    def _check_initialization(self):
        if self.file is None:
            raise Exception("Dataframe associated file cannot be None!")

    def __init__(self, name: str, file: str, description: str, key_column: str = None, group: str = None):
        super().__init__(name, description)
        self.file = file

        self.key_column = key_column
        self.group = group
        self._check_initialization()


class Stage(Enum):
    """
    Enum for specifying the type of data split.
    """
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'


class Benchmark(Base, metaclass=ABCMeta):
    """
    Abstract base class for benchmarks. This class provides the structure for setting up
    benchmarks and retrieving dataframes based on specified parameters.

    Methods:
        set_benchmark_dir(benchmark_dir: str): Sets the directory for the benchmark data.
        set_dataset(dataset: Enum): Abstract method to set the dataset for the benchmark.
        get_dataframe(dataframe: Enum) -> pd.DataFrame: Abstract method to retrieve a dataframe based on the specified Enum.
    """

    def _initialize(self):
        """
        This method contains initialization of the required attributes
        for the benchmark. Subclass should either call it in constructor or
        define them on their own.
        """
        self._benchmark_dir = None
        self._dataset = None
        self._split_id = None
        self._stage = None
        self._metric = None
        self._splits_dir = None

    # Interface required for implementation in the subclass

    @ abstractmethod
    def get_dataframe(self, dataframe: DataFrameDescription):
        pass

    @abstractmethod
    def get_datasets(self):
        pass

    @abstractmethod
    def get_dataframes(self):
        pass

    @abstractmethod
    def get_splits_ids(self):
        """
        Returns the indices of data splits.

        Returns:
            list[Any]: Data splits indices.
        """
        pass

    @abstractmethod
    def get_stages(self):
        return Stage

    @abstractmethod
    def get_metrics(self):
        pass

    # Default methods implementation. Can be overloaded in the subclass.
    # Setting required specifications for generating output data frames
    def set_benchmark_dir(self, benchmark_dir: str):
        self._benchmark_dir = benchmark_dir

    def set_dataset(self, dataset: DatasetDescription) -> None:
        """
        Sets the dataset for the benchmark.

        Args:
            dataset (SingleDRPDataset): The dataset to be used in the benchmark.x
        """
        self._dataset = dataset

    def set_split_id(self, split_id: Any) -> None:
        """
        Sets the split number for data partitioning.

        Args:
            split_number (int): The identifier for the data split.
        """
        self._split_id = split_id

    def set_stage(self, stage: Stage) -> None:
        """
        Sets the type of data split.

        Args:
            stage (SplitType): The type of split (TRAIN, VALIDATION, or TEST).
        """
        self._stage = stage

    def set_metric(self, metric: Enum) -> None:
        """
        Sets the drug response prediction metric.

        Args:
            metric (DRPMetric): The metric to be used for evaluating drug response.
        """
        self._metric = metric

    # Getting data from benchmark

    def get_metric(self) -> Enum:
        return self._metric

    def get_state_string(self) -> str:
        state = '_'.join((str(self._dataset), 'split',
                          str(self._split_id), str(self._stage)))
        return state

    # Checking if all necessary attributes for the benchmark are initialized
    def _check_initialization(self) -> None:
        """
        Checks if all necessary attributes for the benchmark are initialized.

        Raises:
            Exception: If any of the required attributes (_benchmark_dir, _dataset, _split_id, _stage, _metric) are not initialized.
        """
        template_message = "is not specified in benchmark!"
        if self._benchmark_dir is None:
            raise Exception(f"Dataset {template_message}")
        if self._dataset is None:
            raise Exception(f"Dataset {template_message}")
        if self._split_id is None:
            raise Exception(f"Split ID {template_message}")
        if self._stage is None:
            raise Exception(f"Stage {template_message}")
        if self._metric is None:
            raise Exception(f"Metric {template_message}")

     # Optional parameters
    def set_splits_dir(self, splits_dir: str) -> None:
        """
        Setting splits dir is required only if new splits directory is provided.
        New splits directory should be located in the same parent directory as
        the default directory of SingleDRPBenchmark.

        Args:
            splits_dir (str): The path to the directory containing data splits.
        """
        self._splits_dir = splits_dir


class ParameterConverter(Base):

    def __init__(self, string_enum_list: list[DescriptionBase]):
        self._convertion_dict = {}
        for enumeration in string_enum_list:
            enum_dict = {e.name: e for e in enumeration}
            self._convertion_dict.update(enum_dict)

    def str_to_type(self, parameter: str):
        return self._convertion_dict[parameter]

    def update_params(self, params: dict):
        for key, value in params.items():
            if value in self._convertion_dict:
                params[key] = self._convertion_dict[value]
        return params


class DataStager():

    def __init__(self) -> None:
        """
        Initializes the DRPDataStager instance with default values.
        """
        self._out_dir = None
        self._benchmark = None

    def set_benchmark(self, benchmark: Benchmark):
        """
        Sets the benchmark instance to be used for data staging.

        Args:
            benchmark: The benchmark instance to set.
        """
        self._benchmark = benchmark

    def set_output_dir(self, output_dir: str) -> None:
        """
        Sets the output directory where the staged data will be saved.

        Args:
            output_dir (str): The path to the output directory.
        """
        self._out_dir = output_dir

    def stage_experiments(self, datasets: list[DatasetDescription], data_frame_list: list[DataFrameDescription], metric: Enum) -> dict[dict[dict[list[str]]]]:
        """
        Stages all experiments by setting up the necessary directories and paths for each dataset, split, and split type.

        Args:
            datasets (list[DatasetDescription]): List of datasets to stage.
            data_frame_list (list[DataFrameDescription]): List of dataframes to include in each staged dataset.
            metric (DRPMetric): The DRP metric to use for staging.

        Returns:
            dict[dict[dict[str]]]: A dictionary containing paths to the staged data.
        """
        self._check_initialization()
        path_dict = {}
        self._benchmark.set_metric(metric)
        splits_ids = self._benchmark.get_splits_ids()
        for dataset in datasets:
            path_dict[dataset] = {}
            self._benchmark.set_dataset(dataset)
            for split_id in splits_ids:
                path_dict[dataset][split_id] = {}
                self._benchmark.set_split_id(split_id)
                for stage in Stage:
                    self._benchmark.set_stage(stage)

                    sub_dir = self._construct_out_sub_dir(
                        dataset, split_id, stage)

                    # Stub for saving data in fixed format

                    path_dict[dataset][split_id][stage] = []
                    for dataframe_name in data_frame_list:
                        out_file_name = f'{self._benchmark.get_state_string()}_{
                            str(dataframe_name)}.parquet'
                        out_file_path = os.path.join(sub_dir, out_file_name)
                        dataframe = self._benchmark.get_dataframe(
                            dataframe_name)
                        dataframe.to_parquet(out_file_path)
                        path_dict[dataset][split_id][stage].append(
                            out_file_path)
        return path_dict

    def _check_initialization(self) -> None:
        """
        Checks if the necessary attributes for data staging are initialized.

        Raises:
            Exception: If any of the required attributes (_out_dir, _benchmark) are not initialized.
        """
        template_message = "is not initialized!"
        if self._out_dir is None:
            raise Exception(f"Output dir for staging {template_message}")
        if self._benchmark is None:
            raise Exception(f"Benchmark for data staging {template_message}")

    def _construct_out_sub_dir(self, dataset: DatasetDescription, split_id: int, stage: Stage) -> str:
        """
        Constructs the output sub-directory path based on the dataset, split ID, and split type.

        Args:
            single_drp_dataset (SingleDRPDataset): The dataset for which to construct the path.
            split_id (int): The split ID.
            stage (SplitType): The type of split (TRAIN, VALIDATION, TEST).

        Returns:
            str: The constructed sub-directory path.
        """
        path = os.path.join(self._out_dir, dataset.name,
                            f'split_{str(split_id)}', str(stage.value))
        frm.create_outdir(path)
        return path
