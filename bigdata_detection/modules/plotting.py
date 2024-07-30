from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('../definitions')
import definitions_plotting as defplot

start_date = '2024-06-09'
end_date = '2024-06-15'
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
samples_per_day = 431999
days = [(i // samples_per_day) + 1 for i in range(sample_count)]

# print the number of samples and the number of days
print(f"Number of samples: {sample_count}")
print(f"Number of samples per day: {samples_per_day}")
print(f"Number of days: {days[-1]}")

defplot.generateDataPlots(NSsq, Zsq, NSmag, EWmag, Zmag, sample_count, samples_per_day, start_date,end_date)
plt.tight_layout()
plt.show()
