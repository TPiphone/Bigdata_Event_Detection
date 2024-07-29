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
    start_day = int(start_date.split('-')[2])
    end_day = int(end_date.split('-')[2])
    for day in range(start_day, end_day + 1):
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{start_date[:7]}-{day:02d}.{file_type}'
        # print(file_path)
        data += read_txt_file(file_path)
    return data
start_date = '2024-06-09'
end_date = '2024-06-15'
data_mag = get_data('ctumag', read_txt_file, start_date, end_date)
data_squid = get_data('squid', read_txt_file, start_date, end_date)

data_array_mag = process_data(data_mag)
data_array_sq = process_data(data_squid)

time, NSsq, Zsq = parse_squid_data(data_array_sq)
timemag, NSmag, EWmag, Zmag = parse_magnetic_data(data_array_mag)
# print first 5 values of time, NSsq, Zsq
# print(f"First 5 values of array time: {timemag.head()}")
# print(f"First 5 values of array NSmag: {NSmag.head()}")
# print(f"First 5 values of array EWmag: {EWmag.head()}")
# print(f"First 5 values of array Zmag: {Zmag.head()}")

# Calculate the day index for each sample
sample_count = len(time)
samples_per_day = 431999
days = [(i // samples_per_day) + 1 for i in range(sample_count)]

# print the number of samples and the number of days
print(f"Number of samples: {sample_count}")
print(f"Number of samples per day: {samples_per_day}")
print(f"Number of days: {days[-1]}")

def generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day, start_date):
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

    fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True, num='Data Plots')

    # Plot Squid data
    axs[0].plot(NSsq[::10], marker='.', color='orange')
    axs[0].set_title('Squid NS Component')
    axs[0].set_ylabel('NS nT (relative)')

    axs[1].plot(Zsq[::10], marker='.', color='green')
    axs[1].set_title('Squid F Component')
    axs[1].set_ylabel('F nT (relative)')

    # Plot CTU Magnetometer data
    axs[2].plot(NSmag[::10], marker='.', color='orange')
    axs[2].set_title('MAG NS Component')
    axs[2].set_ylabel('NS nT')

    axs[3].plot(EWmag[::10], marker='.', color='blue')
    axs[3].set_title('MAG EW Component')
    axs[3].set_ylabel('EW nT')

    axs[4].plot(Zmag[::10], marker='.', color='green')
    axs[4].set_title('MAG Z Component')
    axs[4].set_ylabel('Z nT')
    axs[4].set_xlabel('Time (s)')

    # Change x-axis labels to dates
    num_ticks = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    tick_positions = np.linspace(-1, sample_count - 1, num_ticks)
    tick_labels = [(pos // samples_per_day) + 1 for pos in tick_positions]
    tick_dates = [(datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=label-1)).strftime('%Y-%m-%d') for label in tick_labels]
    axs[4].set_xticks(tick_positions)
    axs[4].set_xticklabels(tick_dates)
    axs[4].set_xlabel("Date")

generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day, start_date)

plt.tight_layout()
plt.show()
