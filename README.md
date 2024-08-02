#  Space Weather Event Detection Using Geomagnetic Data:A Big Data Approach

 This project covers the complete data engineering process for detection severe space weather events. Using geomagnetic data I will create a model that is able to identity severe weather events (from normal space weather events) in geomagnetic data automatically.

### Installation

To install and run this project, follow these steps:

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/TPiphone/Bigdata_Event_Detection.git
    ```

2. Navigate to the project directory:
    ```
    cd your-repository
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the project:
    ```
    python main.py
    ```

5. You're ready to go! The project is now installed and running on your machine.

### Required Libraries

To use this project, make sure you have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install these libraries by running the following command:

```
pip install numpy pandas matplotlib scikit-learn
```

### Example Usage


Now you can start using the project to detect severe space weather events from geomagnetic data. You can modify the code in `main.py` to customize the detection algorithm or analyze the results further.

For example, you can plot the detected severe weather events using the following code snippet:

```python
import matplotlib.pyplot as plt

# Load the detected events data
events_data = pd.read_csv('events.csv')

# Plot the detected events
plt.plot(events_data['timestamp'], events_data['magnitude'], 'ro')
plt.xlabel('Timestamp')
plt.ylabel('Magnitude')
plt.title('Detected Severe Weather Events')
plt.show()
```

This will generate a plot showing the timestamps and magnitudes of the detected severe weather events.

### Key Features

Here are some key features of this project:

- Automatic detection of severe space weather events from geomagnetic data
- Complete data engineering process for event detection
- Customizable detection algorithm in `main.py`
- Analysis of detected events using `events.csv`
- Plotting of detected events using matplotlib


### Authors and Acknowledgments
- Author - T.R. Perfett
- Supervisor - C.J. Fourie


