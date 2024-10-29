""" subprocess_utils.py """
import logging
import subprocess
from pathlib import Path
from typing import Union, Optional


# def save_subprocess_stdout(
#     result,
#     log_dir: Union[str, Path]='.',
#     log_filename: Optional[str]='logs.txt'):
#     """ Save the captured output from subprocess python package.
#     Args:
#         result: captured output from subprocess python package.
#             E.g. result = subprocess.run(...)
#         log_dir (str or Path): dir to save the logs
#         log_filename (str): file name to save the logs
#     """
#     result_file_name_stdout = log_dir / log_filename
#     with open(result_file_name_stdout, 'w') as file:
#         file.write(result.stdout)
#     return True

def save_subprocess_stdout(
    result: subprocess.CompletedProcess,
    log_dir: Union[str, Path] = '.',
    log_filename: Optional[str] = 'logs.txt',
    mode: str = 'w'  # 'w' for overwrite, 'a' for append
    ) -> bool:
    """Save the captured output from subprocess to a log file.

    Args:
        result (subprocess.CompletedProcess): The result object from subprocess.run().
        log_dir (Union[str, Path]): Directory to save the logs. Defaults to the current directory.
        log_filename (Optional[str]): File name to save the logs. Defaults to 'logs.txt'.
        mode (str): File mode for writing ('w' for overwrite, 'a' for append). Defaults to 'w'.

    Returns:
        bool: True if the output was saved successfully, False otherwise.

    Raises:
        ValueError: If the mode is not 'w' or 'a'.
        IOError: If there is an error writing to the file.
    """
    # Validate the mode
    if mode not in ['w', 'a']:
        raise ValueError("Invalid mode. Use 'w' for overwrite or 'a' for append.")

    # Create the log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    result_file_name_stdout = log_dir / log_filename

    try:
        with open(result_file_name_stdout, mode) as file:
            file.write(result.stdout)
        logging.info(f"Output saved successfully to {result_file_name_stdout}")
        return True
    except IOError as e:
        logging.error(f"Failed to write to {result_file_name_stdout}: {e}")
        return False