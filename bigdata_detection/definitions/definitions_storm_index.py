import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plotDstProxyIndex(hourly_data_quiet):
    plt.figure(figsize=(14, 7))
    plt.plot(hourly_data_quiet.index, hourly_data_quiet['Dst_proxy'], label='Dst Proxy')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Dst Proxy Index for Severe Space Weather')
    plt.xlabel('Time')
    plt.ylabel('Dst Proxy (nT)')
    plt.legend()
    plt.show()