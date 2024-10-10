import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, rfftfreq


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
        df = pd.read_csv(f"/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester_2/Skripsie/Data/NOT_STORM/{day.strftime('%Y-%m-%d')}.csv", index_col=0).dropna()
    except FileNotFoundError:
        # If file is not found in the first directory, try the second one
        try:
            df = pd.read_csv(f"/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester_2/Skripsie/Data/STORM_LABELLED/{day.strftime('%Y-%m-%d')}.csv", index_col=0).dropna()
        except FileNotFoundError:
            # If the file is not found in either directory, print a message and skip this day
            print(f"File not found for {day.strftime('%Y-%m-%d')}. Skipping this day.")
            continue
    
    # Concatenate the current day's data to the combined DataFrame
    combined_df = pd.concat([combined_df, df])
    # print(f"This is the current size of combined_df:", combined_df.shape, " day ", day)
  return combined_df

# Function to create lagged features and shift target variable by 3 hours
def create_lagged_features(data, n_lags=1, shift_steps=900):
    """
    Transform a time series into a supervised learning dataset.
    
    Args:
        data: Multivariate time series (DataFrame or 2D array)
        n_lags: Number of lagged time steps to include as features
        shift_steps: Number of steps to shift the target variable to predict ahead (900 for 3 hours)
    
    Returns:
        X: Feature matrix of lagged values
        Y: Target vector (binary labels 0 or 1) shifted by `shift_steps`
    """
    df = pd.DataFrame(data)
    
    # Create lagged features
    columns = [df.shift(i) for i in range(1, n_lags + 1)]
    columns.append(df.shift(-shift_steps))  # shift target column by shift_steps
    df_supervised = pd.concat(columns, axis=1)
    df_supervised.dropna(inplace=True)  # drop rows with NaN due to shifting
    
    # Split into inputs (X) and outputs (Y), assuming the last column is the target
    X = df_supervised.iloc[:, :-1].values  # all columns except the last one as features
    Y = (df_supervised.iloc[:, -1].values > 0.5).astype(int)  # binary 0/1 labels
    
    return X, Y


def fourier_transform(df, sampling_rate):
    """
    Performs a Fourier transform on a DataFrame, assuming a constant sampling rate.

    Args:
        df: The DataFrame to transform.
        sampling_rate: The sampling rate in samples per second.

    Returns:
        A DataFrame containing the Fourier transform results.
    """

    # Check if the DataFrame has a time index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Create a time index based on the sampling rate
        time_index = pd.date_range(start=0, periods=len(df), freq=f'{1/sampling_rate}s')
        df.index = time_index



    # Perform the Fourier transform on each column
    fft_results = {}
    for column in df.columns:
        fft_data = fft(df[column])
        fft_results[column] = fft_data
    print("GOt to this point!")
    # Create a DataFrame from the Fourier transform results
    fft_df = pd.DataFrame(fft_results)

    # Calculate the frequency axis based on the sampling rate and number of samples
    frequency_axis = np.fft.fftfreq(len(df), 1/sampling_rate)

    # Add the frequency axis as a new index
    fft_df.index = frequency_axis

    return fft_df



   