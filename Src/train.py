from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from explore import import_data_Train
from process import transformation

# Load the training data
data_train = import_data_Train()

# Apply data transformation
data_train_trans = transformation(data_train)

# Split data into features (X) and target variable (y)
X = data_train_trans.drop("Survived", axis=1)
y = data_train_trans["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Instantiate and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict using the trained model
y_pred = log_reg.predict(X_test)

# Print the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print confusion matrix and classification report
print("Logistic Regression - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, y_pred))

# Perform cross-validation and print the scores
scores = cross_val_score(log_reg, X, y, cv=5)
print("Cross-validated scores:", scores)
print("Average score:", scores.mean())
