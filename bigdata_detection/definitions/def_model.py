from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime


def combine_resampled_data(start_date, end_date):
  """Combines resampled data from multiple CSV files into a single DataFrame.

  Args:
    start_date: Start date for combining data.
    end_date: End date for combining data.

  Returns:
    Combined Pandas DataFrame containing resampled data for all days.
  """

  combined_df = pd.DataFrame()
  for day in pd.date_range(start_date, end_date, freq='D'):
    filename = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/MIN DATA/{day.strftime("%Y-%m-%d")}.csv'
    day_df = pd.read_csv(filename, index_col=0, parse_dates=True, dtype=np.float32, skip_blank_lines=True)
    day_df = day_df.dropna()
    combined_df = pd.concat([combined_df, day_df])

  combined_df.sort_index(inplace=True)  # Sort the combined DataFrame by index

  return combined_df