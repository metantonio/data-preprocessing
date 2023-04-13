# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with the following column structure in Excel or .csv file:
# Country   Age     Salary      Purchased
dataset = pd.read_csv('Data.csv') # dataset will be a DataFrame created with pandas library's shortcut 'pd'
# We need a matrix of features (independant variables) wich is gonna be used to predict vector values (dependant variable)
#iloc() is a finder of indexes of rows and columns, parameter can be ranges with :, or just indexes with integers numbers
X = dataset.iloc[:, :-1].values #Matrix of features: contains Country Age and Salary. Taking all rows, and excluding the las column [: , :-1]
y = dataset.iloc[:, -1].values #Vector of values: contains Purchased column. Taking all rows and only the last column [: , -1]
print("X: \n",X) #output: [['France' 44.0 72000.0] ['Spain' 27.0 48000.0] ....]
print("y: \n",y) #output: ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']

# Taking care of missing data
# 1- one aproach is just ignore the row with the missing data
# 2- another aproach is replace the row with the average of data
from sklearn.impute import SimpleImputer #SimpleImputer is a class of sklearn library that will helps us replace missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #instance that wilk take missing values for average (mean)
imputer.fit(X[:, 1:3]) #applying imputer instance to matrix of features between columns Age, Salary (upper range is not taken, that's why is 3)
X[:, 1:3] = imputer.transform(X[:, 1:3]) #replaced matrix of features with transformation of average of missed data
print("updated X: \n",X)

# Encoding categorical data
# Encoding the Independent Variable
# So we should transform countries into numbers, but model could missinterpretate the correlation
# between numbers as importance, that's why we gonna transform column of countries into many columns as countries are listed
# in a binary system, that's called One Hot Encoder. So there a listed 3 diferents countries, we need 3 binary columns 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #transformers parameters need to know wich transform method, what kind of encoding, and indexes of Columns (could be a range). Remainder parameter is requiered to know what to do with others columns
X = np.array(ct.fit_transform(X)) #el método fit will be fiting and transforming the binary columns into the columns that is being transformed. ML models expect arrays, that's why we need a numpy array
print("\n OneHotEncode X: \n",X)


# Encoding the Dependent Variable
# we need transform Yes and No into numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) # will enconde automatically any string in the dependant vector
print("transformed dependant vector: \n",y)

# Splitting the dataset into the Training set and Test set
# we need about 80% of data for training and 20% for testing, data is expected as array
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) #random_state to get same split in not aleatory way
print("X_train: \n",X_train)
print("X_test:\n",X_test)
print("y_train:\n",y_train)
print("y_test:\n",y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)