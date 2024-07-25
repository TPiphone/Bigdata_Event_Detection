import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from definitions import read_txt_file, get_data, process_data, parse_squid_data, parse_magnetic_data, fourier_denoise
from scipy.fftpack import fft, ifft

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
print("This is the diff data frame \n", diff_df)

# Define a threshold for what constitutes a large change
threshold = df.iloc[:, -5:].std() * 0.5
print(threshold)

# Identify large jumps or drops
discontinuities = diff_df[(diff_df > threshold) | (diff_df < -threshold)]
# print("Discontinuities:\n", discontinuities[discontinuities.notnull()])

# if record contains a non Nan value then print the record
print("Records with discontinuities:")
print((df[discontinuities.notnull().any(axis=1)]).count())
