from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime


def combine_resampled_data(start_date, end_date):
  """Combines resampled data from multiple CSV files into a single DataFrame.

  Args:
    start_date: Start date for combining data.
    end_date: End date for combining data.

  Returns:
    Combined Pandas DataFrame containing resampled data for all days.
  """
  combined_df = pd.DataFrame()
  df = pd.DataFrame()
  for day in pd.date_range(start_date, end_date, freq='D'):
    try:
        df = pd.read_csv(f"/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/NOT STORM/{day.strftime('%Y-%m-%d')}.csv", index_col=0).dropna()
    except FileNotFoundError:
        # If file is not found in the first directory, try the second one
        try:
            df = pd.read_csv(f"/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/STORM LABELLED/{day.strftime('%Y-%m-%d')}.csv", index_col=0).dropna()
        except FileNotFoundError:
            # If the file is not found in either directory, print a message and skip this day
            print(f"File not found for {day.strftime('%Y-%m-%d')}. Skipping this day.")
            continue
    
    # Concatenate the current day's data to the combined DataFrame
    combined_df = pd.concat([combined_df, df])
    # print(f"This is the current size of combined_df:", combined_df.shape, " day ", day)
  return combined_df


def create_lagged_features(data, n_lags=1):
    """
    Transform a time series into a supervised learning dataset
    Args:
        data: Multivariate time series (DataFrame or 2D array)
        n_lags: Number of lagged time steps to include as features
    Returns:
        X: Feature matrix of lagged values
        Y: Target vector (binary labels 0 or 1)
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, n_lags + 1)]
    columns.append(df)
    df_supervised = pd.concat(columns, axis=1)
    df_supervised.dropna(inplace=True)

    

    
    # Split into inputs (X) and outputs (Y), assuming last column as binary target
    X = df_supervised.iloc[:, :-1].values  # all columns except the last one as features
    Y = (df_supervised.iloc[:, -1].values > 0.5).astype(int)  # convert to binary 0/1 labels
    return X, Y