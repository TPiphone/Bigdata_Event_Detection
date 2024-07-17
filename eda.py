import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting import read_txt_file, get_data, process_data

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
