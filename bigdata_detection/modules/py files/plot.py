from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('../definitions')
import definitions_plotting as defplot
from datetime import datetime

start_date = '2024-03-29'
end_date = '2024-04-03'

start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
num_days = (end_date_obj - start_date_obj).days

data_mag = defplot.get_data('ctumag', defplot.read_txt_file, start_date, end_date)
data_squid = defplot.get_data('squid', defplot.read_txt_file, start_date, end_date)

data_array_mag = defplot.process_data(data_mag)
data_array_sq = defplot.process_data(data_squid)

time, NSsq, Zsq = defplot.parse_squid_data(data_array_sq)
timemag, NSmag, EWmag, Zmag = defplot.parse_magnetic_data(data_array_mag)
# print first 5 values of time, NSsq, Zsq
# print(f"First 5 values of array time: {timemag.head()}")
# print(f"First 5 values of array NSmag: {NSmag.head()}")
# print(f"First 5 values of array EWmag: {EWmag.head()}")
# print(f"First 5 values of array Zmag: {Zmag.head()}")

# Calculate the day index for each sample
sample_count = len(time)
samples_per_day = int(sample_count / num_days)
print(f"Number of days between {start_date} and {end_date}: {num_days}")
print(f"Number of samples in {num_days} days is : {sample_count}")
print(f"Therefore number of samples per day: {samples_per_day}")

defplot.generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day, start_date,end_date)
plt.tight_layout()
plt.show()
