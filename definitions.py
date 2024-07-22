import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def get_data(file_type, read_txt_file, num_files=3):
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

def process_data(data):
    data_lines = data.strip().split('\n')
    data_array = pd.DataFrame([list(map(float, line.split())) for line in data_lines])
    return data_array

# Squid data
def parse_squid_data(data_array_sq):
    time = data_array_sq[0]
    NSsq = data_array_sq[1]
    Zsq = data_array_sq[2]
    return time, NSsq, Zsq

# CTU Magnetometer data
def parse_magnetic_data(data_array_mag):
    timemag = data_array_mag[0]
    NSmag = data_array_mag[1]
    EWmag = data_array_mag[2]
    Zmag = data_array_mag[3]
    return timemag, NSmag, EWmag, Zmag

# Denoise data using Fourier Transform
def fourier_denoise(data, threshold=0.5):
    fft_coeff = fft(data)
    fft_coeff[np.abs(fft_coeff) < threshold] = 0
    return np.real(ifft(fft_coeff))