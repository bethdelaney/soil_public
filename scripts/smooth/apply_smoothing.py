"""
apply_smoothing.py
Authors: Beth Delaney, Matt Payne
Script to apply Whittaker or Savitzky-Golay smoothing to a timeseries
"""

import logging
import sys
from typing import Optional

import numpy as np
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

    # read the csv into a Pandas DataFrame
    df = pd.read_csv(in_csv_path)

    smoothed_df = smooth_timeseries_sg(df)

    logger.info(smoothed_df)

    return

def smooth_timeseries_sg(df: pd.DataFrame, window_size: int=9, poly_order: int=2) -> pd.DataFrame:
    """
    Author: Beth Delaney

    Apply Savitzky-Golay smoothing to all rows (field-metric pair) in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data (rows are field-metric pairs, columns are dates).
    window_size : int, optional
        The size of the window (must be odd)., by default 9.
    poly_order : int, optional
        The order of the polynomial to fit, by default 2.

    Returns
    -------
    pd.DataFrame
        DataFrame with the smoothed time series.
    """

    # get logger
    logger = logging.getLogger(__name__)

    smoothed_df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original
    for idx, row in df.iterrows():  # Iterate through each row (field-metric pair)
        y = row.values  # Extract the values (time series)
        
        # Print original values for debugging
        logger.info(f"Original {idx} values: {y}")
        
        # Check if the length of the series is greater than or equal to the window size
        if len(y) >= window_size and np.all(np.isfinite(y)):  # Ensure no NaN values in the data
            yhat = savgol_filter(y, window_size, poly_order)
            smoothed_df.loc[idx] = yhat  # Store the smoothed values
            
            # Print smoothed values for verification
            logger.info(f"Smoothed {idx} values: {yhat}")
        else:
            logger.error(f"Skipping smoothing for {idx} because length is less than the window size or contains NaN")
            smoothed_df.loc[idx] = y  # If not enough data or contains NaN, leave it unsmoothed
    return smoothed_df


if __name__ == "__main__":
    main(log_path=sys.argv[1],
         in_csv_path=sys.argv[2],
         out_directory=sys.argv[3])