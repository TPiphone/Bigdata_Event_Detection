import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from definitions import read_txt_file, get_data, process_data, parse_squid_data, parse_magnetic_data, fourier_denoise

data_arr_mag = process_data(get_data('ctumag', read_txt_file))
data_arr_squid = process_data(get_data('squid', read_txt_file))
print(f"Info about mag array {data_arr_mag.info()}")
print(f"This is the squid array {data_arr_squid.info()}")

# Print head of data
print(f"This is the mag head {data_arr_mag.head()}")
print(f"This is the squid head {data_arr_squid.head()}")

# Describe the data
print(f"This is the mag head {data_arr_mag.describe()}")
print(f"This is the squid head {data_arr_squid.describe()}")

# Check for missing values 
print(f"Number of missing values in mag: {data_arr_mag.isnull().sum()}")
print(f"Number of missing values in squid: {data_arr_squid.isnull().sum()}")

# Rolling statistics
print(f"Rolling mean for mag: {data_arr_mag.rolling(window=10).mean()}")
print(f"Rolling std for mag: {data_arr_mag.rolling(window=10).std()}")
print(f"Rolling mean for sq: {data_arr_squid.rolling(window=10).mean()}")
print(f"Rolling std for sq: {data_arr_squid.rolling(window=10).std()}")


# Apply Fourier Transform
# def fourier_transform(data):
#     return np.abs(fft(data_arr_squid))
# ft_squid = fourier_transform(data_arr_squid)

# def plot_fourier_transform(data, title):
#     plt.figure(figsize=(14, 7))
#     plt.plot(data, color='blue')
#     plt.title(title)
#     plt.show()

# plot_fourier_transform(ft_squid, 'Fourier Transform of Squid Data')

# print(f"Lenth of time: {data_arr_squid[0].shape}")
# print(f"Lenth of NS: {data_arr_squid[1].shape}")
# print(f"Lenth of Z: {data_arr_squid[2].shape}")
# print(f"Printing columns {data_arr_squid[1]}")
NSft_sq = fourier_denoise(data_arr_squid[1])
# Zft_sq = fourier_denoise(data_arr_squid[2])

# Plot the data
# plt.figure(figsize=(14, 7))
# plt.subplot(2, 1, 1)
# plt.plot(data_arr_squid[1], label='Original NS')
# plt.plot(NSft_sq, label='Denoised NS (Fourier)', color='red')
# plt.legend()
# plt.title('NS Component - Fourier Transform')

# plt.subplot(2, 1, 2)
# plt.plot(data_arr_squid[2], label='Original Z')
# plt.plot(Zft_sq, label='Denoised Z (Fourier)', color='red')
# plt.legend()
# plt.title('Z Component - Fourier Transform')

# plt.tight_layout()
# plt.show()

