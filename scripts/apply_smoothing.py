"""
apply_smoothing.py
Authors: Beth Delaney, Matt Payne
Script to apply Whittaker or Savitzky-Golay smoothing to a timeseries
"""

import logging
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

def main(log_path: str, in_csv_path: str, out_directory: str, filter: Optional[str]="Savitzky-Golay") -> None:
    """

    Parameters
    ----------
    log_path : str
        the absolute path of the logger to write stdout to.
    in_csv_path : str
        the absolute path of the CSV of spectral values to smooth.
    out_directory : str
        the absolute path of the directory to write files to.
    filter : Optional[str], optional
        the filter to apply, by default "Savitzky-Golay"
    """

    # configures the logger in the namespace
    logging.basicConfig(level=logging.INFO, filename=sys.argv[1], filemode="w", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # get logger
    logger = logging.getLogger(__name__)



if __name__ == "__main__":
    main(log_path=sys.argv[1],
         in_csv_path=sys.argv[2],
         out_directory=sys.argv[3])