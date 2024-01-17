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


# Load training data
data = import_data_Train()

# Display the first few rows to understand data structure
print(data.head())

# Information about data types, missing values, etc.
print(data.info())

# Summary statistics of numerical columns
print(data.describe())

# Count of survivors vs non-survivors
print(data['Survived'].value_counts())

# Pairplot to visualize relationships between features
sns.pairplot(data, hue='Survived')
plt.show()

# Count of survivors vs non-survivors
sns.countplot(x='Survived', data=data)
plt.title('Distribution of Survivors')
plt.show()

# Survival count by gender
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival Distribution by Gender')
plt.show()

# Survival count by passenger class
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Survival Distribution by Passenger Class')
plt.show()
