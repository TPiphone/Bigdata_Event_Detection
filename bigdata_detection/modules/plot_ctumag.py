import matplotlib.pyplot as plt
import numpy as np

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Read XML data from the file
file_path = '/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/2024-06-07.ctumag'
data = read_txt_file(file_path)

# Parse data
lines = data.strip().split('\n')
time = []
NS = []
EW = []
Z = []

for line in lines:
    parts = line.split()
    time.append(float(parts[0]))
    NS.append(float(parts[1]))
    EW.append(float(parts[2]))
    Z.append(float(parts[3]))

# Generate sample indices
sample_count = len(time)
samples = list(range(1, sample_count+1))

# print the number of samples
print('Number of samples:', sample_count)

# Plot the data
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, num='CTU Magnetometer Data')

axs[0].plot(samples, NS, marker='.', color='orange')
axs[0].set_ylabel('NS nT')

axs[1].plot(samples, EW, marker='.', color='blue')
axs[1].set_ylabel('EW nT')

axs[2].plot(samples, Z, marker='.', color='green')
axs[2].set_ylabel('Z nT')
axs[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
