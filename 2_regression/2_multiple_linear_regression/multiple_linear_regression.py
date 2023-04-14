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
print("transformed matrix of features: \n",X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression #this class avoid dummy variable trap (adding dependant variables due to One Hot enconde) in multiple linear regression
regressor = LinearRegression() #this class will choose the best statistical P values that are significant for this model (between [-3:3]), so, don't have to worry for backward eliminatios
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) # show only 2 decimals for any numerical impresion

# We can concatenate vertically and/or horizantlly 2 vectors with np.concatenate. Con reshape podemos mostrar una cierta longitud del vector en vertical en lugar de horizontal
# So, we can comparate the prection vector with the actual values of the test vector to validate our model
# the second parameter of paramenter of .concatenate equal to 1 will make horizontal concatenation
print("comparation: \n",np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making a single prediction of Profit (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print("\nA single prediction: \n",regressor.predict([[1, 0, 0, 160000, 130000, 300000]])) # The One Hot Enconder made [1,0,0] for California

