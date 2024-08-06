# import functions from ../definitions/definitions_EDA
from datetime import datetime, timedelta
import sys
from matplotlib import pyplot as plt
import numpy as np
sys.path.append('../../definitions')
import definitions_EDA as eda
import definitions_plotting as def_plot
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import os
import shutil

start_date = '2024-03-29'
end_date = '2024-04-02'
start_day_control = '2024-03-23'
end_day_control = '2024-03-23'

data_arr_mag = eda.process_data(eda.get_data('ctumag', eda.read_txt_file, start_date, end_date))
data_arr_squid = eda.process_data(eda.get_data('squid', eda.read_txt_file, start_date, end_date))
df = eda.create_dataframe(data_arr_mag, data_arr_squid)
print(f"Data frame shape: \n {df.shape}")

# Create control df
data_arr_mag_cntrl = eda.process_data(eda.get_data('ctumag', eda.read_txt_file, start_day_control, end_day_control))
data_arr_squid_cntrl = eda.process_data(eda.get_data('squid', eda.read_txt_file, start_day_control, end_day_control))
df_cntrl = eda.create_dataframe(data_arr_mag_cntrl, data_arr_squid_cntrl)
print(f"Data frame control shape: \n {df_cntrl.shape}")

discontinuities, threshold, df_cleaned = eda.discontinuity_check(df,df_cntrl)
eda.write_data_summary(df, threshold, discontinuities)
# components, fourier_results = eda.calculate_fourier_transforms(df)
# eda.plot_fourier_transform(fourier_results, components)
print(df_cleaned)

eda.plot_cleaned_data(df_cleaned)