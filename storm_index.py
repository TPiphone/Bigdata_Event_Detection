import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Store the file paths in a list
file_paths = []
start_date = pd.Timestamp('2024-03-21')  # Start date
end_date = pd.Timestamp('2024-03-31')  # End date

# Iterate over the range of dates
for date in pd.date_range(start_date, end_date, freq='D'):
    # Format the file path
    file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{date.strftime("%Y-%m-%d")}.ctumag'
    file_paths.append(file_path)

# Read data from the files and store in the same DataFrame
df = pd.DataFrame()
for file_path in file_paths:
    temp_df = pd.read_csv(file_path, sep="\t", header=None, names=['Time', 'NS', 'EW', 'Z'])
    df = pd.concat([df, temp_df], ignore_index=True)

# Display the DataFrame
print("Concatenated DataFrame:")
print(df.head())

# Calculate the horizontal component of the magnetic field
df['H'] = np.sqrt(df['NS']**2 + df['EW']**2)

# Convert the 'Time' column to datetime format
# Assuming 'Time' is in seconds since the start of the day
df['Time'] = pd.to_datetime(df['Time'], unit='s', origin=start_date)

# Check the first few rows after conversion
print("DataFrame after time conversion:")
print(df.head())

# Set the 'Time' column as the index
df.set_index('Time', inplace=True)

# Resample the data to hourly frequency
hourly_data = df.resample('h').mean()
print("Hourly data shape: ", hourly_data.shape)

# Adjust the time index to start from the default date
default_date = pd.Timestamp('1970-01-01')
date_offset = start_date - default_date
hourly_data.index = hourly_data.index + date_offset

print('New time adjusted columns: \n', hourly_data[['NS', 'EW', 'H']].head())

# Define quiet period for baseline calculation
quiet_period = hourly_data.loc['2024-03-21':'2024-03-27']
H_baseline = quiet_period['H'].mean()
hourly_data['Dst_proxy'] = hourly_data['H'] - H_baseline

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(hourly_data.index, hourly_data['Dst_proxy'], label='Dst Proxy')
plt.axhline(0, color='red', linestyle='--')
plt.title('Dst Proxy Index for Severe Space Weather')
plt.xlabel('Time')
plt.ylabel('Dst Proxy (nT)')
plt.legend()
# plt.show()
