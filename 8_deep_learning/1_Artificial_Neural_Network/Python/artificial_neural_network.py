# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf # pip install tensorflow
print(tf.__version__)

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values # Matrix of Features X, look that some columns that do not have impact on result were deleted
y = dataset.iloc[:, -1].values #
print("Matrix of Features X: \n",X)
print("Vector of Values y: \n",y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) # Randomly, Female was encoded as 0, and Male as 1
print("Matrix of Features X with encoded Gender:\n",X)
# One Hot Encoding the "Geography" column (because it's not a binary option, we need One Hot Enconding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') # CHanged the index of column to 1, that is Geography column
X = np.array(ct.fit_transform(X))
print("Matrix of Features X with encoded Geography: \n",X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Split data, 80% training, 20% testing

# Feature Scaling (always implement with ANN)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential() # With TensorFlow 2.0, Keras is now a module of it.

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # relu: linear rectifier function. Number of units is experimentation until get the best accuracy

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Sigmoid function, it's probabilistic. So, can be more useful than Thereshold function that returns 1 or 0. softmax if it is a non-binary classification

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer: 'adam' best to stocastic gradient descent (better for avoid local minimun problems)
# loss: for binary classication 'binary_crossentropy', other way 'categorical_crossentropy'
# metrics: there are a lot, but most important is accuracy

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# batch_size: every 32 results prediction will be compared with 32 real one
# epochs: number of round to train the ANN

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""
prediction_client = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print("\nProbability of client to leave: ",prediction_client)
print("\nTrue or False that client will leave: ",prediction_client > 0.5)
"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print("predictions of test results:\n",np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n",cm)
accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nacurracy with test dataset: ", accuracy)