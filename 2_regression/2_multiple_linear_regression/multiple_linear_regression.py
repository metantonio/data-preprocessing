# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# 50_Startups.csv are the 50 startups with data of expensives and incomes at differents states, and want to analyze behavior and how to maximize incomes
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') #remember change the index of column if there are a categorical data
X = np.array(ct.fit_transform(X)) #on multiple linear regression we don't need apply scaling features
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression #this class avoid dummy variable trap (adding dependant variables due to One Hot enconde) in multiple linear regression
regressor = LinearRegression() #this class will choose the best statistical P values that are significant for this model (between [-3:3]), so, don't have to worry for backward eliminatios
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))