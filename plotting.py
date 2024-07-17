import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def get_data(file_type, read_txt_file, num_files=7):
    """
    Retrieves data from multiple files and concatenates them into a single string.

    Parameters:
    - file_type (str): The type of file to read (e.g., 'txt', 'csv', etc.).
    - read_txt_file (function): A function that reads a text file and returns its contents.

    Returns:
    - data (str): The concatenated data from all the files.

    """
    data = ''
    for int in range(1, num_files + 1):
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/2024-06-0{int}.{file_type}'
        data += read_txt_file(file_path)
    return data
data_mag = get_data('ctumag', read_txt_file)
data_squid = get_data('squid', read_txt_file)

def process_data(data):
    data_lines = data.strip().split('\n')
    data_array = pd.DataFrame([list(map(float, line.split())) for line in data_lines])
    return data_array

data_array_mag = process_data(data_mag)
data_array_sq = process_data(data_squid)


# Squid data
def parse_squid_data(data_array_sq):
    time = data_array_sq[0]
    NSsq = data_array_sq[1]
    Zsq = data_array_sq[2]
    return time, NSsq, Zsq

time, NSsq, Zsq = parse_squid_data(data_array_sq)

# CTU Magnetometer data
def parse_magnetic_data(data_array_mag):
    timemag = data_array_mag[0]
    NSmag = data_array_mag[1]
    EWmag = data_array_mag[2]
    Zmag = data_array_mag[3]
    return timemag, NSmag, EWmag, Zmag

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

def generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day):
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

    Returns:
    None
    """

    fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True, num='Data Plots')

    # Plot Squid data
    axs[0].plot(NSsq, marker='.', color='orange')
    axs[0].set_title('Squid NS Component')
    axs[0].set_ylabel('NS nT (relative)')

    axs[1].plot(Zsq, marker='.', color='green')
    axs[1].set_title('Squid F Component')
    axs[1].set_ylabel('F nT (relative)')

    # Plot CTU Magnetometer data
    axs[2].plot(NSmag, marker='.', color='orange')
    axs[2].set_title('MAG NS Component')
    axs[2].set_ylabel('NS nT')

    axs[3].plot(EWmag, marker='.', color='blue')
    axs[3].set_title('MAG EW Component')
    axs[3].set_ylabel('EW nT')

    axs[4].plot(Zmag, marker='.', color='green')
    axs[4].set_title('MAG Z Component')
    axs[4].set_ylabel('Z nT')
    axs[4].set_xlabel('Time (s)')

    # Change x-axis labels to days
    num_ticks = 7
    tick_positions = np.linspace(-1, sample_count - 1, num_ticks)
    tick_labels = [(pos // samples_per_day) + 1 for pos in tick_positions]
    axs[4].set_xticks(tick_positions)
    axs[4].set_xticklabels(tick_labels)
    axs[4].set_xlabel("Days")

# generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day)

plt.tight_layout()
# plt.show()
