# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X) #2D array
print("y:\n",y) #1D array
y = y.reshape(len(y),1) #reshaped vertically and into 2D array (10 rows, 1 column)
print("y with reshape:\n",y)

# Feature Scaling (in this model we don't split datase into test and training, bacause we want the correlation of all data)
from sklearn.preprocessing import StandardScaler
#features and result are in a very diferent scale and outside of [-3, 3] range, not dummy variables and as we are not in linear models, we need scale
sc_X = StandardScaler() # scaler that calculates mean and standard deviation of matrix of features
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y) #expect 2D array
print("X after scaling:\n",X)
print("y after scaling:\n",y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf is the gaussian RBF kernel for non linear (dataset) kernel. RBF is the most used
regressor.fit(X, y)

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()