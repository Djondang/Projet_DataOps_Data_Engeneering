import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Charger le premier fichier contenant les données d'entraînement
train_data = pd.read_csv('train.csv')

# Charger le deuxième fichier contenant les données pour la prédiction
pred_data = pd.read_csv('test.csv')

# Diviser les données d'entraînement en variables indépendantes et dépendantes
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Prétraitement des données avec un pipeline
numeric_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Construction du modèle avec un classificateur RandomForest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Entraînement du modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédictions sur le nouvel ensemble pour la prédiction
y_pred = model.predict(pred_data)

# Ajouter les prédictions au dataframe
pred_data['Predicted_Survived'] = y_pred


# Affichage des prédictions
print('Un Aperçu de la prédiction')
print(pred_data[['PassengerId', 'Predicted_Survived']])

