import candle_lib
import pandas as pd
import pyarrow as pa
import h5py



class DataInputOutput:
    def __init__(self, input_dir=None, output_dir=None, loader=None, logger=None) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logger
        self.file_path = None
        self.loader = loader

    def load_data(self, file, format=None):
        """Load data from a file."""

        if format == 'CSV':
            df = pd.read_csv(file)
        elif format == 'parquet':
            df = pd.read_parquet(file)    
        elif format == 'hdf5':
            df = pd.read_hdf(file)
        elif format == 'npy':
            df = np.load(file)
        elif format == 'BenchmarkV1':
            self.logger.critical("Not Implemented: Loading BenchmarkV1")
        elif self.loader not None:
            df = self.loader.load_data(file)
        else:
            self.logger.critical("Unknown format: %s", format)
            return None
        return df

        
