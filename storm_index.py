import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Store the file paths in a list
file_paths = []
for i in range(1, 2):
    file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/2024-06-0{i}.ctumag'
    file_paths.append(file_path)

# Read data from the files and store in the same DataFrame
df = pd.DataFrame()
for file_path in file_paths:
    temp_df = pd.read_csv(file_path, sep="	", header=None, names=['Time', 'NS', 'EW', 'Z'])
    df = pd.concat([df, temp_df], ignore_index=True)

# Display the DataFrame
print(df)

# Calculate the horizontal component of the magnetic field
df['H'] = np.sqrt(df['NS']**2 + df['EW']**2)

# Inspect the calculated horizontal component
# print(df[['Time','NS', 'EW', 'H']].head())

# Convert the 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Set the 'Time' column as the index
df.set_index('Time', inplace=True)

# Resample the data to hourly frequency and calculate the mean
data_hourly = df.resample('h').mean()

# Print the hourly average of the magnetic field components
print(data_hourly[['NS', 'EW', 'H']])

# Define the base date (e.g., the specific day you are working with)
base_date = datetime(2024, 6, 1)  # Replace with the actual base date

# Convert the 'Time' column to timedelta and add it to the base date
df['Timestamp'] = df['Time'].apply(lambda x: base_date + timedelta(seconds=x*3600*24))

