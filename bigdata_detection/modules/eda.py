import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import functions from ../definitions/definitions_EDA
import sys
sys.path.append('../definitions')
import definitions_EDA as eda


from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import os
import shutil
data_arr_mag = eda.process_data(eda.get_data('ctumag', eda.read_txt_file))
data_arr_squid = eda.process_data(eda.get_data('squid', eda.read_txt_file))

df = eda.create_dataframe(data_arr_mag, data_arr_squid)
print(f"Data frame head: \n {df}")

# Calculate the difference between consecutive data points for the last 4 columns
diff_df = df.iloc[::].diff()

# Define a threshold for what constitutes a large change
threshold = df.iloc[:, -5:].std() * 1.5

# Identify large jumps or drops
discontinuities = diff_df[(diff_df > threshold) | (diff_df < -threshold)]

# if record contains a non Nan value then print the record
records_with_discontinuities = df[discontinuities.notnull().any(axis=1)]

# Create the output directory if it doesn't exist
output_dir = '../../output/eda_out'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Move the folder to the output directory
# shutil.move('/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Code/Plotting/SANSA plotting/Bigdata_Event_Detection/bigdata_detection/modules/eda.py', output_dir)

# Describe the data
output_file = os.path.join(output_dir, 'output.txt')
with open(output_file, 'w') as file:
    file.write(f"Data Describe:\n{df.describe()}\n")
    file.write(f"\nThis is the head of the df: \n{df.head()}\n")
    file.write(f"\nShape of the df: {df.shape}\n")
    file.write(f"\nNumber of missing values in the df:\n {df.isnull().sum()}\n") 
    file.write(f"\nThe discontinuity threshold is: \n{threshold}\n")
    file.write(f"\nNumber of discontinuities in the df:\n {discontinuities.notnull().sum()}\n")
    file.write("\nRecords with discontinuities:\n")
    file.write(f"{records_with_discontinuities}\n")
    # Close the file
    file.close()


# Rolling statistics only print the second 10 values
# print(f"Rolling mean for mag:\n {data_arr_mag.rolling(window=10).mean().iloc[10:]}")
# print(f"Rolling std for mag:\n {data_arr_mag.rolling(window=10).std().iloc[10:]}")
# print(f"Rolling mean for sq:\n {data_arr_squid.rolling(window=10).mean().iloc[10:]}")
# print(f"Rolling std for sq:\n {data_arr_squid.rolling(window=10).std().iloc[10:]}")

# Detrend the data
# detrended_df = df.copy()
# for column in detrended_df.columns:
#     if column != 'Time':
#         detrended_df[column] = signal.detrend(detrended_df[column])

# Normalize the data
# normalized_df = (detrended_df - detrended_df.mean()) / detrended_df.std()

# Apply Fourier Transform
sampling_frequency = 5  # 5 measurements per second
components = ['Time', 'NS_SQUID', 'F_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']


# Apply Fourier Transform to each component
fourier_results = {}
for component in components[1:]:
    frequencies, fourier_transform = eda.calculate_fourier_transform(df[component], sampling_frequency)
    fourier_results[component] = (frequencies, fourier_transform)

eda.plot_fourier_transform(fourier_results, components)
