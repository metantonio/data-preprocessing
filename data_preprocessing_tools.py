# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with the following column structure in Excel or .csv file:
# Country   Age     Salary      Purchased
dataset = pd.read_csv('Data.csv') # dataset will be a DataFrame created with pandas library's shortcut 'pd'
# We need a matrix of features (independant variables) wich is gonna be used to predict vector values (dependant variable)
#iloc() is a finder of indexes of rows and columns
X = dataset.iloc[:, :-1].values #Matrix of features: contains Country Age and Salary. Taking all rows, and excluding the las column [: , :-1]
y = dataset.iloc[:, -1].values #Vector of values: contains Purchased column. Taking all rows and only the last column [: , -1]
print("X: \n",X)
print("y: \n",y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)