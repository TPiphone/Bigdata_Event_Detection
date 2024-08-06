
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