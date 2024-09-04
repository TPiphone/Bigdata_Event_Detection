from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import ifft
from statsmodels.tsa.stattools import adfuller
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import gc
from io import StringIO
import time


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        df = pd.read_csv(StringIO(data), sep = "	")
    return df

# def get_data(read_txt_file, start_date, end_date):
#     """
#     Retrieves data from multiple files and concatenates them into a single string.

#     Parameters:
#     - file_type (str): The type of file to read (e.g., 'txt', 'csv', etc.).
#     - read_txt_file (function): A function that reads a text file and returns its contents.
#     - start_date (str): The start date in the format 'YYYY-MM-DD'.
#     - end_date (str): The end date in the format 'YYYY-MM-DD'.

#     Returns:
#     - data (str): The concatenated data from all the files.
#     """
#     data = ''
#     df = pd.DataFrame
#     date_range = pd.date_range(start=start_date, end=end_date, freq='D')

#     for single_date in date_range:
#         file_path_ctu = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{single_date.strftime("%Y-%m-%d")}.ctumag'
#         file_path_squ = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{single_date.strftime("%Y-%m-%d")}.squid'
#         # file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/DUMMY/{single_date.strftime("%Y-%m-%d")}.{file_type}'
#         # print(file_path)
#         try:
#             df_ctu = read_txt_file(file_path_ctu)
#             df_squ = read_txt_file(file_path_squ)
#             df_ctu = df_ctu.drop(columns=[df_ctu.columns[0]])
#             combined_df = pd.concat([df_squ, df_ctu], axis=1)
#             # print(combined_df.head())
#             new_df = create_dataframe(combined_df, single_date)
#             df = df.append(new_df)
#             del df_ctu, df_squ, combined_dßf, new_df
#             gc.collect()
#         except FileNotFoundError:
#             print(f"File not found: {file_path_squ} or {file_path_ctu}")
#             continue
#     return df

def get_data(read_txt_file, start_date, end_date):
    """
    Retrieves data from multiple files and concatenates them into a single DataFrame.

    Parameters:
    - read_txt_file (function): A function that reads a text file and returns its contents as a DataFrame.
    - start_date (str): The start date in the format 'YYYY-MM-DD'.
    - end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
    - df (DataFrame): The concatenated data from all the files.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_list = []

    for single_date in date_range:
        file_path_ctu = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{single_date.strftime("%Y-%m-%d")}.ctumag'
        file_path_squ = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{single_date.strftime("%Y-%m-%d")}.squid'
        
        try:
            df_ctu = read_txt_file(file_path_ctu)
            df_ctu = df_ctu.drop(columns=[df_ctu.columns[0]])
            df_squ = read_txt_file(file_path_squ)
            combined_df = pd.concat([df_squ, df_ctu], axis=1)
            new_df = create_dataframe(combined_df, single_date)
            df_list.append(new_df)
        except FileNotFoundError:
            print(f"File not found for date: {single_date.strftime('%Y-%m-%d')}")
    
    # Combine all the dataframes into one
    if df_list:
        df = pd.concat(df_list, ignore_index=False)
    else:
        df = pd.DataFrame()

    return df


def create_dataframe(df, start_date):
    """
    Creates a DataFrame from magnetometer and SQUID data arrays and sets up the time index.

    Parameters:
    - data_arr_mag (pd.DataFrame): DataFrame containing the magnetometer data.
    - data_arr_squid (pd.DataFrame): DataFrame containing the SQUID data.
    - start_date (str or pd.Timestamp): The start date for the time series.

    Returns:
    - df (pd.DataFrame): The resulting DataFrame with time-based indexing.
    """
    components = ['Time', 'NS_SQUID', 'Z_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']
    df.columns = components
   
    df['DateTime'] = df['Time'].apply(lambda x: start_date + timedelta(seconds=x))
    df = df.drop(columns=['Time'])

    # Assuming 'df' is your DataFrame and 'date_column' is the column with date information
    df['DateTime'] = pd.to_datetime(df['DateTime'])  # Convert the column to datetime
    df.index = pd.DatetimeIndex(df['DateTime'])
    # df.set_index('DateTime', inplace=True)  # Set the datetime column as the index


    # Print the row every time the date changes
    # df['Date'] = df['DateTime'].dt.date
    # df['Date_change'] = df['Date'].ne(df['Date'].shift())
    # print(df[df['Date_change']])
    
    # Check for and handle duplicate indices
    # duplicate_indices = df.index[df.index.duplicated()]
    # print(f"The total number of duplicates is: {len(duplicate_indices)}")

    # if len(duplicate_indices) > 0:
    #     # If duplicates are found, increment the duplicates by a small time offset
    #     df.index = df.index + pd.to_timedelta(df.groupby(df.index).cumcount(), unit='ns')

    # Optionally, infer the frequency of the time series and set it
    # inferred_freq = pd.infer_freq(df.index)
    # if inferred_freq:
    #     df.index.freq = inferred_freq
    return df


######################################## CHECKED ##############################################

# Calculate the Fourier Transform for each component
def calculate_fourier_transform(data, sampling_frequency):
    L = len(data)
    fourier_transform = np.fft.fft(data)
    fourier_transform[0] = 0  # Set the first element to 0 to remove the DC component
    frequencies = np.fft.fftfreq(L, 1 / sampling_frequency)
    return frequencies, fourier_transform

# Apply Fourier Transform
def calculate_fourier_transforms(df):
    sampling_frequency = 1  # 5 measurements per second
    components = ['Time', 'NS_SQUID', 'Z_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']

    fourier_results = {}
    for component in components[1:]:
        frequencies, fourier_transform = calculate_fourier_transform(df[component], sampling_frequency)
        fourier_results[component] = (frequencies, fourier_transform)
    return components,fourier_results

# Resample the data to a lower frequency
def resample_data(df, resample_frequency):

    components = ['Time', 'NS_SQUID', 'Z_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']
    for component in components[1:]:
        ser = df[component].squeeze()
        

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Group by the date part of the index and resample within each group
    df_resampled = df.groupby(df.index.date).apply(lambda x: x.resample(resample_frequency).mean())

    # Convert the index back to a DatetimeIndex if necessary
    df_resampled.index = pd.to_datetime(df_resampled.index.get_level_values(1))

    # Drop any rows with NaN values
    df_resampled = df_resampled.dropna()

    return df_resampled


def manual_resample_data(df, resample_frequency, agg_methods=None, interpolate=False):
    """
    Manually resample the DataFrame to a lower frequency.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        resample_frequency (str): The resampling frequency (e.g., 'S', 'D', 'W', 'M').
        agg_methods (dict or str, optional): Aggregation methods for resampling.
                                             Can be a string (e.g., 'mean') or a 
                                             dictionary specifying methods per column.
        interpolate (bool, optional): Whether to interpolate missing values after resampling.

    Returns:
        pd.DataFrame: The resampled DataFrame.
    """
    # Convert index to a datetime index if it is not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Determine how to group the data based on the resample frequency
    if resample_frequency == 's':
        grouped = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour, df.index.minute, df.index.second])
    elif resample_frequency == 'D':
        grouped = df.groupby(df.index.date)
    elif resample_frequency == 'W':
        grouped = df.groupby(df.index.to_period('W').start_time)
    elif resample_frequency == 'M':
        grouped = df.groupby(df.index.to_period('M').start_time)
    elif resample_frequency == 'Q':
        grouped = df.groupby(df.index.to_period('Q').start_time)
    elif resample_frequency == 'Y':
        grouped = df.groupby(df.index.to_period('Y').start_time)
    else:
        raise ValueError("Unsupported resample frequency. Use 'S', 'D', 'W', 'M', 'Q', or 'Y'.")

    # Apply the aggregation method or default to mean if not provided
    if agg_methods is None:
        agg_methods = 'mean'

    if isinstance(agg_methods, str):
        # If a single method is provided as a string, apply it to all columns
        df_resampled = grouped.agg(agg_methods)
    elif isinstance(agg_methods, dict):
        # If a dictionary is provided, apply the specified methods to each column
        df_resampled = grouped.agg(agg_methods)
    else:
        raise ValueError("agg_methods must be a string or a dictionary")

    if interpolate:
        df_resampled = df_resampled.interpolate()

    # Print the number of dropped rows due to missing values
    missing_before_dropping = df_resampled.isnull().sum().sum()
    print(f"Number of missing values before dropping rows: {missing_before_dropping}")
    
    df_resampled = df_resampled.dropna()
    
    missing_after_dropping = df_resampled.isnull().sum().sum()
    print(f"Number of missing values after dropping rows: {missing_after_dropping}")

    return df_resampled




def plot_fourier_transform(fourier_results, components):
    # Plot the results
    plt.figure(figsize=(14, 10))

    for i, component in enumerate(components[1:], 1):
        frequencies, fourier_transform = fourier_results[component]
        plt.subplot(3, 2, i)
        # plt.plot(np.abs(fourier_transform[:len(fourier_transform) * np.abs(fourier_transform[:len(fourier_transform)//2]))
        # plt.plot(frequencies, np.abs(fourier_transform))
        plt.title(f'Fourier Transform of {component}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.yscale('log')  
        plt.xscale('log')
        # plt.xlim(0, max(frequencies[:len(frequencies)//2]))  # Set x-axis limit to fit the data

        # plt.xlim(0, max(frequencies[:len(frequencies)//2]))  # Set x-axis limit to fit the data

    plt.tight_layout()
    # plt.show()

def perform_dickey_fuller_test(df):
    df_results = {}
    for column in df.columns:
        df_results[column] = dickey_fuller_test(pd.Series(df[column]))

# Display the results
    for column, result in df_results.items():
        print(f"Dickey-Fuller Test for {column}:")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("\n")

def remove_outliers(df):
    iqr = df.quantile(0.75) - df.quantile(0.25)
    q3  = df.quantile(0.75)
    q1  = df.quantile(0.25)
    upper = q3 + (2.5 * iqr)
    # print(f"Upper limit:\n {upper}")
    upper_array = df >= upper

    lower = q1 - (2.5 * iqr)
    # print(f"Lower limit:\n {lower}")
    lower_array = df <= lower
    total = upper_array.sum() + lower_array.sum()
    new_df = df[(df < upper) & (df > lower)]
    new_df = new_df.dropna()
    # new_df = new_df.interpolate(method="time") 
    removed_values_index = df.index.difference(new_df.index)
    # print("Index of removed values:", removed_values_index)
    # print the number of outiers removed
    print(f"Number of outliers removed:\n {total}")
    return new_df

def z_score_test(df, threshold_z=3):
    # Calculate the Z-scores for each column (excluding the index)
    z = np.abs(stats.zscore(df.iloc[:, 1:]))  # Assumes the first column is to be excluded
    
    # Find outlier indices
    outlier_indices = np.where(z > threshold_z)
    
    # Identify rows with any outlier and drop them
    rows_to_drop = np.unique(outlier_indices[0])
    removed_outliers_df = df.drop(df.index[rows_to_drop])
    
    # Print Z-scores
    print(z)
    
    # Print the number of outliers removed
    print(f"Number of outliers removed: {len(df) - len(removed_outliers_df)}")
    
    return removed_outliers_df

# def denoise_df(df):
#     denoised_df = fft_denoiser(value, 0.001, True) 
#     plt.plot(time, google_stock['Open'][0:300]) 
#     plt.plot(time, denoised_google_stock_price) 
#     plt.xlabel('Date', fontsize = 13) 
#     plt.ylabel('Stock Price', fontsize = 13) 
#     plt.legend([‘Open’,’Denoised: 0.001']) 
#     plt.show()
#     return denoise_df

def plot_seasonal_decompose(df):
    components = ['Time', 'NS_SQUID', 'Z_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, component in enumerate(components[1:], 1):
        df[f'{component}_trend'].plot(ax=axes[i // 2, i % 2], color='blue')
        axes[i // 2, i % 2].set_title(f'{component} Trend')
        axes[i // 2, i % 2].set_ylabel('Amplitude')
        axes[i // 2, i % 2].set_xlabel('Time')
        axes[i // 2, i % 2].grid(False)

        df[f'{component}_seasonal'].plot(ax=axes[i // 2, i % 2], color='red')
        axes[i // 2, i % 2].set_title(f'{component} Seasonal')
        axes[i // 2, i % 2].set_ylabel('Amplitude')
        axes[i // 2, i % 2].set_xlabel('Time')
        axes[i // 2, i % 2].grid(False)

        df[f'{component}_residual'].plot(ax=axes[i // 2, i % 2], color='green')
        axes[i // 2, i % 2].set_title(f'{component} Residual')
        axes[i // 2, i % 2].set_ylabel('Amplitude')
        axes[i // 2, i % 2].set_xlabel('Time')
        axes[i // 2, i % 2].grid(False)

    plt.tight_layout()
    plt.show()




# Describe the data, 
def write_data_summary(df, threshold, discontinuities):
    # Create the output directory if it doesn't exist
    output_dir = '../../output/eda_out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'output.txt')
    #print("Connecteda successfully!")
    with open(output_file, 'w') as file:
        file.write(f"Data Describe:\n{df.describe()}\n")
        file.write(f"\nThis is the head of the df: \n{df.head()}\n")
        file.write(f"\nShape of the df: {df.shape}\n")
        file.write(f"\nNumber of missing values in the df:\n {df.isnull().sum()}\n") 
        file.write(f"\nThe discontinuity threshold is: \n{threshold}\n")
        file.write(f"\nNumber of discontinuities in the df:\n {discontinuities.shape[0]}\n")
        file.write("\nRecords with discontinuities:\n")
        file.write(f"{discontinuities}\n")
    # Close the file
        file.close()

# def check_missing_time(df):
#     complete_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='200ms')  # 200 ms for 5 Hz
#     # Reindex the DataFrame to the complete time index
#     df_reindexed = df.reindex(complete_time_index)
#     # Identify missing time steps
#     missing_time_steps = df_reindexed[df_reindexed.isna().any(axis=1)]
#     proportion_missing = len(missing_time_steps) / len(complete_time_index)
#     print("Proportion of missing time steps:", proportion_missing)



def discontinuity_check(df):
    # threshold =  df.mean()+ (df.std()*1.5)
    # times_gaps = df.index - df.index.shift(1)
    # return times_gaps, threshold
    # Calculate the difference between consecutive data points to identify large jumps or drops
    diff_df = df.iloc[:].diff().dropna()
    print(f"Diff_df: \n {diff_df}")

    # Define a threshold for what constitutes a large change
    threshold =  df.mean()+ (df.std()*1.5)
    # print(f"Threshold for each column: \n {threshold}")

    # Find the records that contain discontinuities
    discontinuities = diff_df.abs() > threshold
    # print(f"Discontinuities: \n {discontinuities}")
    # Get the records with discontinuities
    records_with_discontinuities = df[discontinuities.any(axis=1)]
    # print the number of discontinuities
    print(f"Number of discontinuities: {records_with_discontinuities.shape[0]}")
    # Remove the records with discontinuities
    df_cleaned = df[~discontinuities.any(axis=1)]

    return records_with_discontinuities, threshold, df_cleaned 



def is_continuous(series):
    id_first_true = (series > 0).idxmax()
    id_last_true = (series > 0)[::-1].idxmax()
    return all((series > 0).loc[id_first_true:id_last_true] == True) 


# Detrend the data
# def detrend_data(df):
#     detrended_df = df.copy()
#     for column in detrended_df.columns:
#         if column != 'Time':
#             detrended_df[column] = signal.detrend(detrended_df[column])

def smooth_data(df):
    df_interpolated = df.ffill().bfill()
    window_size = 5
    # df_smoothed = df_interpolated.rolling(window=window_size, min_periods=1).mean()
    return df_interpolated

def plot_cleaned_data(df_cleaned):
    """
    Plots each column of the cleaned DataFrame with a different color.
    
    Parameters:
    - df_cleaned (pd.DataFrame): The DataFrame to plot.
    """
    
    # df_smoothed = smooth_data(df_cleaned)
    df_smoothed = df_cleaned
    num_columns = df_smoothed.shape[1]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # List of colors to use for each column
    fig, axes = plt.subplots(num_columns, 1, figsize=(10, num_columns * 3))
    
    if num_columns == 1:
        axes = [axes]
    
    for i, column in enumerate(df_smoothed.columns):
        data_array = df_smoothed[column].values  # Create a separate array without a time index
        axes[i].plot(data_array, color=colors[i % len(colors)])  # Use a different color for each column
        axes[i].set_title(f'{column} Data Array')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel(column)
        axes[i].grid(False)
    
    plt.tight_layout()
    plt.show()

def generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day, start_date, end_date):
    """
    Generates data plots for Squid and CTU Magnetometer data.

    Parameters:
    NSsq (array-like): Array of NS Squid data.
    Zsq (array-like): Array of Z Squid data.
    NSmag (array-like): Array of NS Magnetometer data.
    EWmag (array-like): Array of EW Magnetometer data.
    Zmag (array-like): Array of Z Magnetometer data.
    sample_count (int): Total number of samples.
    samples_per_day (int): Number of samples per day.
    start_date (str): The start date in the format 'YYYY-MM-DD'.

    Returns:
    None
    """

    fig, axs = plt.subplots(5, 1, figsize=(16, 10), sharex=True, num='Data Plots')

    # Plot Squid data
    
    axs[0].plot(NSsq, marker='.', color='red', linewidth = 0.5)
    axs[0].set_title('Squid NS Component')
    axs[0].set_ylabel('NS nT (relative)')
    # axs[0].axhline(y=100, color='black', linestyle='--')

    axs[1].plot(Zsq, marker='.', color='blue', linewidth = 0.5)
    axs[1].set_title('Squid Z Component')
    axs[1].set_ylabel('Z nT (relative)')

    # Plot CTU Magnetometer data
    axs[2].plot(NSmag, marker='.', color='orange', linewidth = 0.5)
    axs[2].set_title('MAG NS Component')
    axs[2].set_ylabel('NS nT')

    axs[3].plot(EWmag, marker='.', color='purple', linewidth = 0.5)
    axs[3].set_title('MAG EW Component')
    axs[3].set_ylabel('EW nT')

    axs[4].plot(Zmag, marker='.', color='green', linewidth = 0.5)
    axs[4].set_title('MAG Z Component')
    axs[4].set_ylabel('Z nT')
    axs[4].set_xlabel('Data Points')

    # Change x-axis labels to dates
    # num_ticks = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    # tick_positions = np.linspace(-1, sample_count - 1, num_ticks)
    # tick_labels = [(pos // samples_per_day) + 1 for pos in tick_positions]
    # tick_dates = [(datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=label)).strftime('%Y-%m-%d') for label in tick_labels]
    # axs[4].set_xticks(tick_positions)
    # axs[4].set_xticklabels(tick_dates)
    # axs[4].set_xlabel("Date")

def dickey_fuller_test(series):
    """
    Perform the Augmented Dickey-Fuller test.

    Parameters:
    - series (pd.Series): The time series data to test.

    Returns:
    - dict: A dictionary with the test results.
    """
    result = adfuller(series)
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations Used': result[3],
        'Critical Values': result[4],
        'IC Best': result[5]
    }


def test_stationarity(df):
    decomposed_results = {}  # Store decomposition results for each column
    
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Skipping non-numeric column: {column}")
            continue
        
        print(f"Analyzing column: {column}")
        
        # Visual Inspection
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column], label=f'{column} Time Series', color='blue')
        plt.title(f'Time Series Data - {column}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Seasonal Decomposition
        print(f"Decomposing the time series for column: {column}")
        try:
            result = seasonal_decompose(df[column].dropna(), model='additive', period=None)  # Adjust period if known
            decomposed_results[column] = result  # Store the result
            
            # Plot the decomposed components
            result.plot()
            plt.suptitle(f'Seasonal Decomposition - {column}', fontsize=16)
            plt.show()

        except Exception as e:
            print(f"Error decomposing {column}: {e}")
            continue

        # Access the trend, seasonal, and residual components if needed
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
        print(f"Trend:\n{trend.dropna().head()}")
        print(f"Seasonal:\n{seasonal.dropna().head()}")
        print(f"Residual:\n{residual.dropna().head()}")

    return decomposed_results


def detect_spikes_and_correct(df, column_name, threshold=15):
    """
    Detects spikes in the data and smooths them by adjusting the spike value.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to check for spikes.
    - threshold (float): The threshold for detecting spikes in terms of standard deviations.

    Returns:
    - corrected_df (pd.DataFrame): The DataFrame with the spikes corrected.
    """

    # Ensure the DataFrame index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    # Calculate the difference between consecutive values
    diff = df[column_name].diff()

    # Detect spikes by finding where the difference exceeds the threshold
    # std_dev = diff.std()
    spikes = diff.abs() > threshold # * std_dev
    # print(spikes)
    # Get the indices where spikes occur
    spike_indices = spikes[spikes].index
    print(spike_indices)
    if len(spike_indices) == 0:
        print("No spikes detected.")
        return df

    # print(f"Spikes detected at indices: {spike_indices}")
    print(f" Found ",len(spike_indices), "spikes")
    corrected_df = df.copy()

    for index in spike_indices:
        # Find the positions before and after the spike
        prev_index = corrected_df.index.get_loc(index) - 7500
        next_index = corrected_df.index.get_loc(index) + 7500
 
        # my_window = next_index - prev_index
        my_window = 4000


        smoothed_values = corrected_df.iloc[prev_index:next_index + 1, corrected_df.columns.get_loc(column_name)].rolling(window=my_window, min_periods=1).mean()
        smoothed_values = gaussian_filter1d(smoothed_values, sigma=2000)
        corrected_df.iloc[prev_index:next_index + 1, corrected_df.columns.get_loc(column_name)] = smoothed_values
        # Apply moving average only between prev_index and next_index
        # corrected_df.iloc[prev_index:next_index, corrected_df.columns.get_loc(column_name)] = corrected_df.iloc[prev_index:next_index, corrected_df.columns.get_loc(column_name)].rolling(window=my_window).mean()

    return corrected_df

def calculate_mean_of_five(df):
    # Initialize an empty DataFrame to store the means
    df_means = pd.DataFrame()

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Initialize a list to hold the means for the current column
        column_means = []
        
        # Iterate through the DataFrame in steps of 5
        for i in range(0, len(df), 15):
            # Get the mean of the current group of 5 numbers
            group_mean = df[column].iloc[i:i+15].mean()
            # Append the mean to the list for the current column
            column_means.append(group_mean)
        
        # Add the list of means to the new DataFrame as a new column
        df_means[column] = column_means

    return df_means