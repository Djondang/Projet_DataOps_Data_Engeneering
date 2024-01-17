import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Function to import training data
def import_data_Train() -> pd.DataFrame:
    """
    Import CSV file as a DataFrame (Training Data)
    Output: data [pd.DataFrame]
    """
    data = pd.read_csv("Data/train.csv")
    return data


# Function to import test data
def import_data_Test() -> pd.DataFrame:
    """
    Import CSV file as a DataFrame (Test Data)
    Output: data [pd.DataFrame]
    """
    data = pd.read_csv("Data/test.csv")
    return data