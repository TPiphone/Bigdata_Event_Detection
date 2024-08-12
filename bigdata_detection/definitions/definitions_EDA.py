from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import ifft
from statsmodels.tsa.stattools import adfuller


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def get_data(file_type, read_txt_file, start_date, end_date):
    """
    Retrieves data from multiple files and concatenates them into a single string.

    Parameters:
    - file_type (str): The type of file to read (e.g., 'txt', 'csv', etc.).
    - read_txt_file (function): A function that reads a text file and returns its contents.
    - start_date (str): The start date in the format 'YYYY-MM-DD'.
    - end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
    - data (str): The concatenated data from all the files.
    """
    data = ''
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for single_date in date_range:
        # file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{single_date.strftime("%Y-%m-%d")}.{file_type}'
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/DUMMY/{single_date.strftime("%Y-%m-%d")}.{file_type}'
        # print(file_path)
        try:
            data += read_txt_file(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
    
    return data

def process_data(data):
    data_lines = data.strip().split('\n')
    data_array = pd.DataFrame([list(map(float, line.split())) for line in data_lines])
    return data_array

def create_dataframe(data_arr_mag, data_arr_squid, start_date):
    components = ['Time', 'NS_SQUID', 'F_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']
    mag_data_without_time = data_arr_mag.drop(columns=[0])  
    df = pd.concat([data_arr_squid, mag_data_without_time], axis=1)
    df.columns = components
    df['Time'] = pd.to_datetime(start_date) + pd.to_timedelta(df['Time'], unit='s')
    df.set_index('Time', inplace=True)  # Set the 'Time' column as the index
    return df

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
    components = ['Time', 'NS_SQUID', 'F_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']

    fourier_results = {}
    for component in components[1:]:
        frequencies, fourier_transform = calculate_fourier_transform(df[component], sampling_frequency)
        fourier_results[component] = (frequencies, fourier_transform)
    return components,fourier_results

# Resample the data to a lower frequency
def resample_data(df, resample_frequency):
    df_resampled = df.resample(resample_frequency).mean()
    # print(f"Resampled data frame shape: \n {df_resampled.shape}")
    print(f"Resampled data frame head: \n {df_resampled.head()}")
    # df_resampled = df_resampled.ffill().bfill()
    return df_resampled

def plot_fourier_transform(fourier_results, components):
    # Plot the results
    plt.figure(figsize=(14, 10))

    for i, component in enumerate(components[1:], 1):
        frequencies, fourier_transform = fourier_results[component]
        plt.subplot(3, 2, i)
        plt.plot(frequencies[:len(frequencies)//2], 2.0/len(fourier_transform) * np.abs(fourier_transform[:len(fourier_transform)//2]))
        # plt.plot(frequencies, np.abs(fourier_transform))
        plt.title(f'Fourier Transform of {component}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.yscale('log')  
        plt.xscale('log')
        # plt.xlim(0, max(frequencies[:len(frequencies)//2]))  # Set x-axis limit to fit the data

    plt.tight_layout()
    # plt.show()


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



def discontinuity_check(df,df_cntrl):
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
    axs[0].plot(NSsq, marker='.', color='red')
    axs[0].set_title('Squid NS Component')
    axs[0].set_ylabel('NS nT (relative)')

    axs[1].plot(Zsq, marker='.', color='blue')
    axs[1].set_title('Squid F Component')
    axs[1].set_ylabel('F nT (relative)')

    # Plot CTU Magnetometer data
    axs[2].plot(NSmag, marker='.', color='orange')
    axs[2].set_title('MAG NS Component')
    axs[2].set_ylabel('NS nT')

    axs[3].plot(EWmag, marker='.', color='purple')
    axs[3].set_title('MAG EW Component')
    axs[3].set_ylabel('EW nT')

    axs[4].plot(Zmag, marker='.', color='green')
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


