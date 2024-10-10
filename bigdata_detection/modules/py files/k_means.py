import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import sys
sys.path.append('../definitions')
import def_k_means as k_means
# from plot_squid import read_txt_file, get_data

# data_mag = get_data('ctumag', read_txt_file)
# data_squid = get_data('squid', read_txt_file)

data_mag = k_means.get_data('ctumag', k_means.read_txt_file,1)

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