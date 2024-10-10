import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../definitions')
import definitions_storm_index as sti

# Store the file paths in a list
file_paths = []
start_date_check = pd.Timestamp('2024-05-21')  # Start date
end_date_check = pd.Timestamp('2024-05-26')  # End date
start_date_quiet = pd.Timestamp('2024-03-21')  # Start date
end_date_quiet = pd.Timestamp('2024-03-31')  # End date

df_check = sti.read_data(start_date_check, end_date_check)
df_quiet = sti.read_data(start_date_quiet, end_date_quiet)
print("This is the head for df check:\n", df_check.head())
print("This is the head for df quiet:\n", df_quiet.head())

# Calculate the number of days for each dataset
num_days_check = (end_date_check - start_date_check).days + 1
num_days_quiet = (end_date_quiet - start_date_quiet).days + 1
print("Number of days for df check:", num_days_check)
print("Number of days for df quiet:", num_days_quiet)

# Resample the data to hourly frequency
hourly_data_check = df_check.resample('h').mean()
hourly_data_quiet = df_quiet.resample('h').mean()

# Define quiet period for baseline calculation
H_baseline = hourly_data_quiet['H'].mean()
hourly_data_quiet['Dst_proxy'] = hourly_data_quiet['H'] - H_baseline

# Plotting
sti.plotDstProxyIndex(hourly_data_quiet)
