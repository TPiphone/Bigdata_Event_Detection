import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Store the file paths in a list
file_paths = []
start_date_check = pd.Timestamp('2024-05-21')  # Start date
end_date_check = pd.Timestamp('2024-05-26')  # End date
start_date_quiet = pd.Timestamp('2024-03-21')  # Start date
end_date_quiet = pd.Timestamp('2024-03-31')  # End date

def read_data(start_date, end_date):

    # Store the file paths in a list
    file_paths = []

    # Iterate over the range of dates
    for date in pd.date_range(start_date, end_date, freq='D'):
        # Format the file path
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/{date.strftime("%Y-%m-%d")}.ctumag'
        # print(file_path)
        file_paths.append(file_path)

    # Read data from the files and store in the same DataFrame
    df = pd.DataFrame()
    for file_path in file_paths:
        temp_df = pd.read_csv(file_path, sep="\t", header=None, names=['Time', 'NS', 'EW', 'Z'])
        df = pd.concat([df, temp_df], ignore_index=True)

    # Calculate the horizontal component of the magnetic field
    df['H'] = np.sqrt(df['NS']**2 + df['EW']**2)

    # Convert the 'Time' column to datetime format
    # Assuming 'Time' is in seconds since the start of the day
    df['Time'] = pd.to_datetime(df['Time'], unit='s', origin=start_date)

    # Set the 'Time' column as the index
    df.set_index('Time', inplace=True)
    return df

df_check = read_data(start_date_check, end_date_check)
df_quiet = read_data(start_date_quiet, end_date_quiet)
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
plt.figure(figsize=(14, 7))
plt.plot(hourly_data_quiet.index, hourly_data_quiet['Dst_proxy'], label='Dst Proxy')
plt.axhline(0, color='red', linestyle='--')
plt.title('Dst Proxy Index for Severe Space Weather')
plt.xlabel('Time')
plt.ylabel('Dst Proxy (nT)')
plt.legend()
plt.show()
