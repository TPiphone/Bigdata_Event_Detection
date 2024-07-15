import matplotlib.pyplot as plt
import numpy as np

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def get_data(file_type, read_txt_file):
    """
    Retrieves data from multiple files and concatenates them into a single string.

    Parameters:
    - file_type (str): The type of file to read (e.g., 'txt', 'csv', etc.).
    - read_txt_file (function): A function that reads a text file and returns its contents.

    Returns:
    - data (str): The concatenated data from all the files.

    """
    data = ''
    for int in range(1, 8):
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/2024-06-0{int}.{file_type}'
        data += read_txt_file(file_path)
    return data
data_mag = get_data('ctumag', read_txt_file)
data_squid = get_data('squid', read_txt_file)


# Parse data
# Squid data
if data_squid:
    lines = data_squid.strip().split('\n')
    time = []
    NS = []
    Z = []
    for line in lines:
        parts = line.split()
        time.append(float(parts[0]))
        NS.append(float(parts[1]))
        Z.append(float(parts[2]))

# CTU Magnetometer data
if data_mag:
    lines = data_mag.strip().split('\n')
    #time = []
    NS = []
    EW = []
    Z = []
    for line in lines:
        parts = line.split()
        #time.append(float(parts[0]))
        NS.append(float(parts[1]))
        EW.append(float(parts[2]))
        Z.append(float(parts[3]))


# Calculate the day index for each sample
sample_count = len(time)
samples_per_day = 431999
days = [(i // samples_per_day) + 1 for i in range(sample_count)]

# print the number of samples and the number of days
print(f"Number of samples: {sample_count}")
print(f"Number of days: {days[-1]}")

# Generate sample indices
samples = list(range(1, sample_count+1))


fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True, num='Data Plots')

# Plot Squid data
axs[0].plot( NS, marker='.', color='orange')
axs[0].set_title('Squid NS Component')
axs[0].set_ylabel('NS nT (relative)')

axs[1].plot( Z, marker='.', color='green')
axs[1].set_title('Squid F Component')
axs[1].set_ylabel('F nT (relative)')

# Plot CTU Magnetometer data
axs[2].plot( NS, marker='.', color='orange')
axs[2].set_title('MAG NS Component')
axs[2].set_ylabel('NS nT')

axs[3].plot( EW, marker='.', color='blue')
axs[3].set_title('MAG EW Component')
axs[3].set_ylabel('EW nT')

axs[4].plot( Z, marker='.', color='green')
axs[4].set_title('MAG Z Component')
axs[4].set_ylabel('Z nT')
axs[4].set_xlabel('Time (s)')

# Change x-axis labels to days
num_ticks = 7
tick_positions = np.linspace(-1, sample_count - 1, num_ticks)
tick_labels = [(pos // samples_per_day) +1 for pos in tick_positions]
axs[4].set_xticks(tick_positions)
axs[4].set_xticklabels(tick_labels)
axs[4].set_xlabel("Days")

plt.tight_layout()
plt.show()
