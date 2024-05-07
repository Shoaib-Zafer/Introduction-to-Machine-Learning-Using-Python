import os  # Import the os module for operating system functionality
import tarfile  # Import the tarfile module for working with tar archives
from six.moves import urllib  # Import urllib from the six library for compatibility
import pandas as pd

# Define constants for downloading housing data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")  # Define the path to store housing datasets
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"  # Define the URL to download housing data

# Define a function to fetch housing data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):  # Check if the housing data directory doesn't exist
        os.makedirs(housing_path)  # Create the housing data directory if it doesn't exist

    tgz_path = os.path.join(housing_path, "housing.tgz")  # Define the path to save the downloaded tgz file
    urllib.request.urlretrieve(housing_url, tgz_path)  # Download the tgz file from the URL
    housing_tgz = tarfile.open(tgz_path)  # Open the downloaded tgz file
    housing_tgz.extractall(path=housing_path)  # Extract the contents of the tgz file to the housing data directory
    housing_tgz.close()  # Close the tgz file



# Function to load housing data from a CSV file into a Pandas DataFrame
def load_housing_data(housing_path=HOUSING_PATH):
    # Create the path to the housing CSV file
    csv_path = os.path.join(housing_path, "housing.csv")
    
    # Read the CSV file into a Pandas DataFrame
    return pd.read_csv(csv_path)

fetch_housing_data()
df= load_housing_data()
print(df)