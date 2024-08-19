from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{single_date.strftime("%Y-%m-%d")}.{file_type}'
        # file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/DUMMY/{single_date.strftime("%Y-%m-%d")}.{file_type}'
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
    print(f"Data frame head: \n {data_array.head()}")
    print(f"Data frame shape: {data_array.shape}")
    return data_array

# Squid data
def parse_squid_data(data_array_sq):
    time = data_array_sq[0]
    NSsq = data_array_sq[1]
    Zsq = data_array_sq[2]
    return time, NSsq, Zsq

# CTU Magnetometer data
def parse_magnetic_data(data_array_mag):
    timemag = data_array_mag[0]
    NSmag = data_array_mag[1]
    EWmag = data_array_mag[2]
    Zmag = data_array_mag[3]
    return timemag, NSmag, EWmag, Zmag

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
    axs[0].plot(NSsq[::5], marker='.', color='red')
    axs[0].set_title('Squid NS Component')
    axs[0].set_ylabel('NS nT (relative)')

    axs[1].plot(Zsq[::5], marker='.', color='blue')
    axs[1].set_title('Squid F Component')
    axs[1].set_ylabel('F nT (relative)')

    # Plot CTU Magnetometer data
    axs[2].plot(NSmag[::5], marker='.', color='orange')
    axs[2].set_title('MAG NS Component')
    axs[2].set_ylabel('NS nT')

    axs[3].plot(EWmag[::5], marker='.', color='purple')
    axs[3].set_title('MAG EW Component')
    axs[3].set_ylabel('EW nT')

    axs[4].plot(Zmag[::5], marker='.', color='green')
    axs[4].set_title('MAG Z Component')
    axs[4].set_ylabel('Z nT')
    axs[4].set_xlabel('Time (s)')

    # Change x-axis labels to dates
    num_ticks = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    tick_positions = np.linspace(-1, sample_count - 1, num_ticks)
    tick_labels = [(pos // samples_per_day) + 1 for pos in tick_positions]
    tick_dates = [(datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=label)).strftime('%Y-%m-%d') for label in tick_labels]
    axs[4].set_xticks(tick_positions)
    axs[4].set_xticklabels(tick_dates)
    axs[4].set_xlabel("Date")