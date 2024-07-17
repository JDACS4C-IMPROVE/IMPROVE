import candle
import pandas as pd
import pyarrow as pa
import h5py
import logging
import numpy as np
# import tensorflow as tf
# import torch


class DataInputOutput:

    # Default format
    FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
    logging.basicConfig(format=FORMAT)

    def __init__(self, input_dir=None, output_dir=None, loader=None, logger=None, framework=None) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger('IO')
        self.framework = framework
        self.file_path = None
        self.loader = loader

        # import tensorflow or pytorch module for framework specific operations
        if framework == 'tensorflow':
            try:
                import tensorflow as tf
                # importlib.import_module(tf, package=tensorflow)
                self.tf = tf
            except ImportError:
                self.tf = None
                self.logger.warning("Tensorflow not installed")
        elif framework == 'pytorch':
            try:
                import torch
                self.torch = torch
            except ImportError:
                self.torch = None
                self.logger.warning("Pytorch not installed")

    def set_input_dir(self, input_dir):
        self.input_dir = input_dir

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_logger(self, logger):
        self.logger = logger

    def set_loader(self, loader):
        self.loader = loader

    def load_data(self, file, format=None, loader=None):
        """Load data from a file."""

        if format == 'CSV':
            df = pd.read_csv(file)
        elif format == 'parquet':
            df = pd.read_parquet(file)
        elif format == 'hdf5':
            df = pd.read_hdf(file)
        elif format == 'npy':
            df = np.load(file, allow_pickle=True)
        elif format == 'BenchmarkV1':
            self.logger.critical("Not Implemented: Loading BenchmarkV1")
        elif not self.loader is None:
            df = self.loader.load_data(file)
        else:
            self.logger.critical("Unknown format: %s", format)
            return None
        return df

    def load_features(self, file, format=None, loader=None):
        """Load features from a file."""
        return self.load_data(file, format, loader)

    def load_labels(self, file, format=None, loader=None):
        """Load labels from a file."""
        return self.load_data(file, format, loader)

    def save_weights(self, file, data, format=None, loader=None):
        """Save tensorflow or pytorch weights to a file."""

        # Save model weights for tensorflow or pytorch
        if format == 'tensorflow':
            data.save(file) or data.save_weights(file)
        elif format == 'pytorch':
            torch.save(data, file)
        elif not self.loader is None:
            self.loader.save_weights(file, data)
        else:
            self.logger.critical("Unknown format: %s", format)
            return None

    def load_weights(self, file, format=None, loader=None):
        """Load tensorflow or pytorch weights from a file."""

        # Load model weights for tensorflow or pytorch
        if format == 'tensorflow':
            model = tf.keras.models.load_model(file)
        elif format == 'pytorch':
            model = torch.load(file)
        elif not self.loader is None:
            model = self.loader.load_weights(file)
        else:
            self.logger.critical("Unknown format: %s", format)
            return None
        return model

    def save_data(self, file, data, format=None, loader=None):
        """Save data to a file. Data is a pandas dataframe or numpy array."""

        # if data is a numpy array convert it to a pandas dataframe
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # if data is a tensorflow or pytorch tensor convert it to a dataframe
        # if self.framework == 'tensorflow':
        #     if isinstance(data, tf.Tensor):
        #         data = pd.DataFrame(data.numpy())
        # elif self.framework == 'pytorch':
        #     if isinstance(data, torch.Tensor):
        #         data = pd.DataFrame(data.numpy())

        if format == 'CSV':
            data.to_csv(file)
        elif format == 'parquet':
            data.to_parquet(file)
        elif format == 'hdf5':
            data.to_hdf(file, key='data')
        elif format == 'npy':
            np.save(file, data)
        elif format == 'BenchmarkV1':
            self.logger.critical("Not Implemented: Saving BenchmarkV1")
        elif not self.loader is None:
            self.loader.save_data(file, data)
        else:
            self.logger.critical("Unknown format: %s", format)
            return None
        return df


if __name__ == "__main__":
    io = DataInputOutput(framework='tensorflow')
    df = io.load_data("./tmp/iris.csv", format="CSV")
    print(df)
    io.save_data("./tmp/iris.parquet", df, format="parquet")
    # io.save_data("./tmp/iris.hdf5", df, format="hdf5")
    io.save_data("./tmp/iris.npy", df, format="npy")

    print(io.load_data("./tmp/iris.parquet", format="parquet"))
    # print(io.load_data("./tmp/iris.hdf5", format="hdf5"))
    print(io.load_data("./tmp/iris.npy", format="npy"))

    # print(io.load_data("./tmp/iris.csv", loader=candle.CandleLoader()))
    # print(io.load_data("./tmp/iris.csv", format="BenchmarkV1"))
    # print(io.save_data("./tmp/iris.csv", io.load_data("./tmp/iris.csv", format="CSV"), format="CSV"))
    # print(io.save_data("./tmp/iris.parquet", io.load_data("./tmp/iris.parquet", format="parquet"), format="parquet"))
    # print(io.save_data("./tmp/iris.hdf5", io.load_data("./tmp/iris.hdf5", format="hdf5"), format="hdf5"))
    # print(io.save_data("./tmp/iris.npy", io.load_data("./tmp/iris.npy", format="npy"), format="npy"))
    # print(io.save_data("./tmp/iris.csv", io.load_data("./tmp/iris.csv", loader=candle.CandleLoader())))
