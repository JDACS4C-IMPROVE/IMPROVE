"""
This module provides a Timer class for measuring elapsed time, displaying the duration, and saving runtime information to a file.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union


class Timer:
    """A class to measure elapsed time."""

    def __init__(self):
        """Initialize the timer and record the start time."""
        self.start = time.time()

    def timer_end(self):
        """Calculate the elapsed time since the timer started."""
        self.end = time.time()
        self.time_diff = self.end - self.start
        self.hours = int(self.time_diff // 3600)
        self.minutes = int((self.time_diff % 3600) // 60)
        self.seconds = self.time_diff % 60
        self.time_diff_dict = {'hours': self.hours,
                               'minutes': self.minutes,
                               'seconds': self.seconds}

    def display_timer(self, print_fn=print) -> Dict:
        """Display the elapsed time in a formatted string.

        Args:
            print_fn (callable): Function to use for printing the elapsed time.
        """
        self.timer_end()
        tt = self.time_diff_dict
        print(f"Elapsed Time: {self.hours:02}:{self.minutes:02}:{self.seconds:05}")
        return self.time_diff_dict

    def save_timer(self,
                   dir_to_save: Union[str, Path]='.',
                   filename: str='runtime.json',
                   extra_dict: Optional[Dict]=None) -> bool:
        """Save the runtime information to a JSON file.

        Args:
            dir_to_save (Union[str, Path]): Directory to save the JSON file.
            filename (str): Name of the JSON file.
            extra_dict (Optional[Dict]): Additional data to include in the JSON file.

        Returns:
            bool: True if the operation was successful.
        """
        if isinstance(extra_dict, dict):
            self.time_diff_dict.update(extra_dict)
        with open(Path(dir_to_save) / filename, 'w') as json_file:
            json.dump(self.time_diff_dict, json_file, indent=4)
        return True