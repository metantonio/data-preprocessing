# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(len(dataset)):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) #sabía de antemano que habían 20 columnas

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
#parameters:
## transactions: need transactions list
## min_support: % of product repeated at least 3 times per day in 1 week = 3*7/7500 = 0.003
## min_confidence: % of rules that you have confidence, start in 0.8 and going down until have some relevants rules.
## min_lift: 3
## min_length: Number of elements in the left side that need to be to make a rule
## max_length: Number of elementos in the right side given N products at the left side.

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
#results
print("results rules:\n",results)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
print("non sorted:\n",resultsinDataFrame)

## Displaying the results sorted by descending lifts
print("sorted:\n",resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))