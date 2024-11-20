"""
apply_smoothing.py
Authors: Beth Delaney, Matt Payne
Script to apply Whittaker or Savitzky-Golay smoothing to a timeseries
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
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

    plot_comparison(df, smoothed_df, out_directory)

    return

def smooth_timeseries_sg(df: pd.DataFrame, window_size: int=9, poly_order: int=2) -> pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing to each column (except date) in the DataFrame.

    Original Author: Beth Delaney.
    Amended by Matt Payne to reflect the column-wise nature of the index CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data (columns are spectral indices, rows are timestamps).
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

    # sort by datetime, just to be sure
    df.sort_values(by="datetime", ascending=True, inplace=True)

    smoothed_df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original

    # get only the spectral indices columns
    spectral_cols = df.filter(["NDVI", "SAVI", "NBR", "NDWI"], axis="columns")

    # for each column, do:
    for col in spectral_cols:
        
        # get the values as a numpy array
        y = df[col].values

        logger.info(f"Original {col} values: {y}")
        
        # Check if the length of the series is greater than or equal to the window size
        if len(y) >= window_size and np.all(np.isfinite(y)):  # Ensure no NaN values in the data
            yhat = savgol_filter(y, window_size, poly_order)
            smoothed_df[col] = yhat  # Store the smoothed values
            
            # Print smoothed values for verification
            logger.info(f"Smoothed {col} values: {yhat}")
        else:
            logger.error(f"Skipping smoothing for {col} because length is less than the window size or contains NaN")
            # If not enough data or contains NaN, leave it unsmoothed

    return smoothed_df

def plot_comparison(df: pd.DataFrame, smoothed_df: pd.DataFrame, out_directory: str) -> None:
    """
    Plots a comparision graph per index of the original values and the smoothed values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data (columns are spectral indices, rows are timestamps).
    smoothed_df : pd.DataFrame
        DataFrame containing the smoothed time series data (columns are spectral indices, rows are timestamps).
    out_directory : str
        the absolute path of the directory to write files to.
    """

    # get the logger
    logger = logging.getLogger(__name__)

    # convert date and time to dates for x axis
    dates = pd.to_datetime(df["datetime"])

    # get only the spectral indices columns
    spectral_cols = df.filter(["NDVI", "SAVI", "NBR", "NDWI"], axis="columns")

    # loop through each index
    for spectral_index in spectral_cols:

        original_index = df[spectral_index]
        smoothed_index = smoothed_df[spectral_index]

        # plot object
        plt.figure(figsize=(10, 6))

        # scatter plot for 'Raw' values
        plt.scatter(dates, original_index, label=f"Original {spectral_index}", marker="o", color="blue", s=3) # s controls the size of the markers
        plt.plot(dates, smoothed_index, label=f"Smoothed {spectral_index}", color="orange")

        # adding labels and title
        plt.xlabel("Date")
        plt.ylabel(spectral_index)
        plt.title(f"Original and Smoothed {spectral_index}")
        plt.legend()

        # cleanup xaxis for dates
        locator = mdates.AutoDateLocator(minticks=(len(dates) * 0.02), maxticks=len(dates) * 0.1)  # have min and max date ticks proportional to the length of the timeseries

        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # ensure date is formatted as YYYY-MM-DD
        plt.gcf().autofmt_xdate(rotation=45)

        # show the plot
        plt.tight_layout()
        plt.savefig(os.path.join(out_directory, f"{spectral_index}.png"))
        plt.show()

    return

if __name__ == "__main__":
    main(log_path=sys.argv[1],
         in_csv_path=sys.argv[2],
         out_directory=sys.argv[3])