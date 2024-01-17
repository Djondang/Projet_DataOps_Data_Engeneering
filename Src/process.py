from explore import import_data_Train, import_data_Test

# Load training and test data files
data_train = import_data_Train()  # Loading training data
data_test = import_data_Test()  # Loading test data


def transformation(data):
    # Dropping unnecessary columns from the dataset
    data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1, inplace=True)

    # Encoding 'Sex' column to numerical values
    data['Sex'] = data['Sex'].replace({'male': 1, 'female': 0})
    print(data.head())  # Displaying the first few rows of the transformed data

    # Checking missing values before filling
    missing_values = data.isnull().sum()
    print("Missing values before filling:", missing_values)

    # Filling missing values in the 'Age' column with the mean value
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # Checking missing values after filling
    missing_values_after_fill = data['Age'].isnull().sum()
    print("Missing values after filling:", missing_values_after_fill)

    # Displaying the filled 'Age' column
    print("Filled 'Age' column:", data['Age'])

    return data  # Returning the transformed data


transformation(data_train)  # Applying the transformation function to the training data
