import argparse
import logging
import os
import random
import sys
from logging import Logger
from typing import Dict, List

import numpy as np

from .file_utils import get_file


def fetch_file(
    link: str, subdir: str, unpack: bool = False, md5_hash: str = None
) -> str:
    """
    Convert URL to file path and download the file if it is not already
    present in spedified cache.

    :param string link: URL of the file to download
    :param string subdir: Local path to check for cached file.
    :param bool unpack: Flag to specify if the file to download should be decompressed too. \
        (default: False, no decompression)
    :param string md5_hash: MD5 hash used as a checksum to verify data integrity. \
        Verification is carried out if a hash is provided. \
        (default: None, no verification)

    :return: local path to the downloaded, or cached, file.
    :rtype: string
    """

    fname = os.path.basename(link)
    return get_file(
        fname, origin=link, unpack=unpack, md5_hash=md5_hash, cache_subdir=subdir
    )


def verify_path(path: str) -> None:
    """
    Verify if a directory path exists locally. If the path does not exist,
    but is a valid path, it recursivelly creates the specified directory path
    structure.

    :param string path: Description of local directory path
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(
    logfile: str,
    logger: Logger,
    verbose: bool = False,
    fmt_line: str = "[%(asctime)s %(process)d] %(message)s",
    fmt_date: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Set up the event logging system. Two handlers are created. One to send
    log records to a specified file and one to send log records to the
    (defaulf) sys.stderr stream. The logger and the file handler are set to
    DEBUG logging level. The stream handler is set to INFO logging level, or to
    DEBUG logging level if the verbose flag is specified. Logging messages
    which are less severe than the level set will be ignored.

    :param string logfile: File to store the log records
    :param Logger logger: Python object for the logging interface
    :param boolean verbose: Flag to increase the logging level from INFO to DEBUG. \
        It only applies to the stream handler.
    """
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter(fmt_line, datefmt=fmt_date))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(""))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)


# REFORMATING UTILS


def eval_string_as_list(str_read: str, separator: str, dtype) -> List:
    """Parse a string and convert it into a list of lists.

    Parameters
    ----------
    str_read : string
        String read (from configuration file or command line, for example)
    separator : character
        Character that specifies the separation between the lists
    dtype : data type
        Data type to decode the elements of the list

    Return
    ----------
    decoded_list : list
        List extracted from string and with elements of the
        specified type.
    """

    # Figure out desired type
    ldtype = dtype
    if ldtype is None:
        ldtype = np.int32

    # Split list
    decoded_list = []
    out_list = str_read.split(separator)

    # Convert to desired type
    for el in out_list:
        decoded_list.append(ldtype(el))

    return decoded_list


def eval_string_as_list_of_lists(
    str_read: str, separator_out: str, separator_in: str, dtype
) -> List:
    """Parse a string and convert it into a list of lists.

    Parameters
    ----------
    str_read : string
        String read (from configuration file or command line, for example)
    separator_out : character
        Character that specifies the separation between the outer level lists
    separator_in : character
        Character that specifies the separation between the inner level lists
    dtype : data type
        Data type to decode the elements of the lists

    Return
    ----------
    decoded_list : list
        List of lists extracted from string and with elements of the specified type.
    """

    # Figure out desired type
    ldtype = dtype
    if ldtype is None:
        ldtype = np.int32

    # Split outer list
    decoded_list = []
    out_list = str_read.split(separator_out)
    # Split each internal list
    for line in out_list:
        in_list = []
        elem = line.split(separator_in)
        # Convert to desired type
        for el in elem:
            in_list.append(ldtype(el))
        decoded_list.append(in_list)

    return decoded_list


def str2bool(v: str) -> bool:
    """
    This is taken from:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-
    argparse Because type=bool is not interpreted as a bool and
    action='store_true' cannot be undone.

    :param string v: String to interpret

    :return: Boolean value. It raises and exception if the provided string cannot \
        be interpreted as a boolean type.

        - Strings recognized as boolean True : \
            'yes', 'true', 't', 'y', '1' and uppercase versions (where applicable).
        - Strings recognized as boolean False : \
            'no', 'false', 'f', 'n', '0' and uppercase versions (where applicable).
    :rtype: boolean
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def keras_default_config() -> Dict:
    """
    Defines parameters that intervine in different functions using the keras
    defaults.

    This helps to keep consistency in parameters between frameworks.
    """

    kerasDefaults = {}

    # Optimizers
    # kerasDefaults['clipnorm']=?               # Maximum norm to clip all parameter gradients
    # kerasDefaults['clipvalue']=?              # Maximum (minimum=-max) value to clip all parameter gradients
    kerasDefaults["decay_lr"] = 0.0  # Learning rate decay over each update
    kerasDefaults["epsilon"] = 1e-8  # Factor to avoid divide by zero (fuzz factor)
    kerasDefaults[
        "rho"
    ] = 0.9  # Decay parameter in some optmizer updates (rmsprop, adadelta)
    kerasDefaults[
        "momentum_sgd"
    ] = 0.0  # Momentum for parameter update in sgd optimizer
    kerasDefaults[
        "nesterov_sgd"
    ] = False  # Whether to apply Nesterov momentum in sgd optimizer
    kerasDefaults[
        "beta_1"
    ] = 0.9  # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults[
        "beta_2"
    ] = 0.999  # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults["decay_schedule_lr"] = 0.004  # Parameter for nadam optmizer

    # Initializers
    kerasDefaults[
        "minval_uniform"
    ] = -0.05  # Lower bound of the range of random values to generate
    kerasDefaults[
        "maxval_uniform"
    ] = 0.05  # Upper bound of the range of random values to generate
    kerasDefaults["mean_normal"] = 0.0  # Mean of the random values to generate
    kerasDefaults[
        "stddev_normal"
    ] = 0.05  # Standard deviation of the random values to generate

    return kerasDefaults


def set_seed(seed: int) -> None:
    """
    Set the seed of the pseudo-random generator to the specified value.

    :param int seed: Value to intialize or re-seed the generator.
    """
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(seed)

    random.seed(seed)
