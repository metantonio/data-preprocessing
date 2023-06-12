# XGBoost (works for regression models and classification models)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data_regression.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# y_train must be encoded in a newer update XGBoost model before training it
""" from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train) """

# Training XGBoost on the Training set
from xgboost import XGBRegressor #use XGBRegressor for regressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)

print("revisión linea 27")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
print(y_pred)
r2 = r2_score(y_test, y_pred)
print("r2 of polynomial regression: ", r2)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print("Accuracy after 10 folds: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))