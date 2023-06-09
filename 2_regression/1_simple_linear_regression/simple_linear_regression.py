# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # adjustment/fit method to train de model with linear regression

# Predicting the Test set results (return a vector prediction)
y_pred = regressor.predict(X_test) #for the test will try to predict salary based on number of years experience
print("X_test: \n", X_test)
print("y_prediction: \n",y_pred) # is returning points over the line because is a line actually

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red') #red points will be associated with the real observation data
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #blue line is the regression prediction of ML. In simple linear regression will be the same use X_train or X_test
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
#look that 'predict' method always expects a 2D array as the format of its inputs, that's why [[12]]
print(regressor.predict([[12]])) #output aproximated is $ 138531,5

# Getting the final linear regression equation with the values of the coeffcients
m = regressor.coef_
b = regressor.intercept_
print(m)
print(b)
print("equation: \n Salary = ",m[0]," * YearsExperience + ",b)

