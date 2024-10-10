# import functions from ../definitions/definitions_EDA
from datetime import datetime, timedelta
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
sys.path.append('../definitions')
import definitions_EDA as eda
# import definitions_plotting as def_plot
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import shutil
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Date variables
start_date = '2024-03-20'
end_date = '2024-03-21'

# Import data
data_arr_mag = eda.process_data(eda.get_data('ctumag', eda.read_txt_file, start_date, end_date))
data_arr_squid = eda.process_data(eda.get_data('squid', eda.read_txt_file, start_date, end_date))

# Create dataframe
df = eda.create_dataframe(data_arr_mag, data_arr_squid, start_date)
print(f' Head of dataframe: \n',df.head())
print(f' \n Shape of df', df.shape)
print(f'\nTypes for each column: \n', df.dtypes)

# Plot raw data
eda.generateDataPlots(df['NS_SQUID'].values, df['F_SQUID'].values, df['NS_Fluxgate'].values, df['EW_Fluxgate'].values, df['Z_Fluxgate'].values, df.shape[0], 300, start_date, end_date)

# check for missing values 

for column in df.columns:
    print(f"\n Number of missing values in {column} is: ", df[column].isna().sum())

# check for outliers

print(f' \n Shape of df before removing outliers', df.shape)
outliers_removed = eda.remove_outliers(df)
print(f' \n Shape of df after removing outliers', outliers_removed.shape)
eda.plot_cleaned_data(outliers_removed)

# Resample data

resampled_df = eda.resample_data(outliers_removed, 's')
print(f'This is the length of the resampled data frame', len(resampled_df))
eda.plot_cleaned_data(resampled_df)

# Test for stationarity

eda.perform_dickey_fuller_test(resampled_df)
df = resampled_df
eda.test_stationarity(df)

# Fourier Transform
components, fourier_results = eda.calculate_fourier_transforms(df)
eda.plot_fourier_transform(fourier_results, components)

