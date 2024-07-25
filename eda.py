import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from definitions import read_txt_file, get_data, process_data, parse_squid_data, parse_magnetic_data, fourier_denoise
from scipy.fft import fft, ifft, fftfreq

data_arr_mag = process_data(get_data('ctumag', read_txt_file))
data_arr_squid = process_data(get_data('squid', read_txt_file))
# print(f"Info about mag array {data_arr_mag.info()}")
# print(f"Info about squid array {data_arr_squid.info()}")

# Print head of data
print(f"This is the mag head \n{data_arr_mag.head()}")
print(f"This is the squid head \n{data_arr_squid.head()}")

# Describe the data
# print(f"Mag Descripbe \n {data_arr_mag.describe()}")
# print(f"Squid Descripbe \n {data_arr_squid.describe()}")

# Check for missing values 
# print(f"Number of missing values in mag:\n {data_arr_mag.isnull().sum()}")
# print(f"Number of missing values in squid:\n {data_arr_squid.isnull().sum()}")

# Rolling statistics only print the second 10 values

# print(f"Rolling mean for mag:\n {data_arr_mag.rolling(window=10).mean().iloc[10:]}")
# print(f"Rolling std for mag:\n {data_arr_mag.rolling(window=10).std().iloc[10:]}")
# print(f"Rolling mean for sq:\n {data_arr_squid.rolling(window=10).mean().iloc[10:]}")
# print(f"Rolling std for sq:\n {data_arr_squid.rolling(window=10).std().iloc[10:]}")


# Create a single dataframe to store the data
components = ['Time', 'NS_SQUID', 'F_SQUID', 'NS_Fluxgate', 'EW_Fluxgate', 'Z_Fluxgate']
mag_data_without_time = data_arr_mag.drop(columns=[0])  
df = pd.concat([data_arr_squid, mag_data_without_time], axis=1)
df.columns = components
print(f"Data frame head: \n {df}")

# Calculate the difference between consecutive data points for the last 4 columns
diff_df = df.iloc[:, -5:].diff()
# print("This is the diff data frame \n", diff_df)

# Define a threshold for what constitutes a large change
threshold = df.iloc[:, -5:].std() * 0.5
# print(threshold)

# Identify large jumps or drops
discontinuities = diff_df[(diff_df > threshold) | (diff_df < -threshold)]
# print("Discontinuities:\n", discontinuities[discontinuities.notnull()])

# if record contains a non Nan value then print the record
# print("Records with discontinuities:")
# print((df[discontinuities.notnull().any(axis=1)]).count())



# Fourier Transform myself
ns_squid = np.array(df['NS_SQUID'])
print(ns_squid)
print(ns_squid.shape)

# # Find the numbers that occur more than once
# unique_elements, counts = np.unique(ns_squid, return_counts=True)
# duplicates = unique_elements[counts > 1]
# print("Numbers that occur more than once:", len(duplicates))
N = len(ns_squid)
SAMPLE_RATE = 5
yf = fft(ns_squid)
xf = fftfreq(N, 1 / SAMPLE_RATE)
plt.plot(xf, np.abs(yf))
plt.show()




# # Apply Fourier Transform
# # Set the sampling frequency
# sampling_frequency = 5  # 5 measurements per second

# # Calculate the Fourier Transform for each component
# def calculate_fourier_transform(data, sampling_frequency):
#     L = len(data)
#     fourier_transform = np.fft.fft(data)
#     frequencies = np.fft.fftfreq(L, 1 / sampling_frequency)
#     # print the first 5 frequencies
#     # print(frequencies[:5])
#     # print(fourier_transform[:5],"\n")
#     return frequencies, fourier_transform

# # Apply Fourier Transform to each component

# fourier_results = {}

# for component in components[1:]:
#     frequencies, fourier_transform = calculate_fourier_transform(df[component], sampling_frequency)
#     fourier_results[component] = (frequencies, fourier_transform)


# # Plot the results
# plt.figure(figsize=(14, 10))

# for i, component in enumerate(components[1:], 1):
#     frequencies, fourier_transform = fourier_results[component]
#     plt.subplot(3, 2, i)
#     plt.plot(frequencies[:len(frequencies)//2], 2.0/len(fourier_transform) * np.abs(fourier_transform[:len(fourier_transform)//2]))
#     plt.title(f'Fourier Transform of {component}')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Amplitude')
#     plt.xlim(0, max(frequencies[:len(frequencies)//2]))  # Set x-axis limit to fit the data

# plt.tight_layout()
# # plt.show()