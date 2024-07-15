import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# from plot_squid import read_txt_file, get_data

# data_mag = get_data('ctumag', read_txt_file)
# data_squid = get_data('squid', read_txt_file)

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def get_data(file_type, read_txt_file, num_files=7):
    """
    Retrieves data from multiple files and concatenates them into a single string.

    Parameters:
    - file_type (str): The type of file to read (e.g., 'txt', 'csv', etc.).
    - read_txt_file (function): A function that reads a text file and returns its contents.

    Returns:
    - data (str): The concatenated data from all the files.

    """
    data = ''
    for int in range(1, num_files + 1):
        file_path = f'/Users/tristan/Library/CloudStorage/OneDrive-StellenboschUniversity/Academics/Final_year/Semester 2/Skripsie/Data/SANSA/2024-06-0{int}.{file_type}'
        data += read_txt_file(file_path)
    return data
data_mag = get_data('squid', read_txt_file,1)

# Parse data into numpy array
data_lines = data_mag.strip().split('\n')
data_array = np.array([list(map(float, line.split())) for line in data_lines])

# Select every 10th data point
data_array = data_array[::10]

# Convert numpy array to pandas DataFrame
df = pd.DataFrame(data_array, columns=['Time', 'NS', 'Z'])
# print(df.head())

# Visualize the Data
# sns.scatterplot(data=df, x='Time', y='NS', hue='Z')
# plt.show()
 
# Normalizing the Data
X_train, X_test, y_train, y_test = train_test_split(df[['Time', 'NS']], df[['Z']], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)



print("All is well with the world")