# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting = 3 to ignore " "

# Cleaning the texts
import re
import nltk # natural preprocessing langugage library, to prevent 'stop words' that do not impact in the review
nltk.download('stopwords')
from nltk.corpus import stopwords # importing downloades stopwords
from nltk.stem.porter import PorterStemmer # stem module, make a steamming process where root of a word is taken, no matter conjugation. Helps to reduce dimensions
corpus = [] # Will be a list with cleaned reviews
for i in range(0, len(dataset)):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #removing al punctuations and replaced by space
  review = review.lower() # Converted to lower case
  review = review.split() # Every work is now an element of a list
  ps = PorterStemmer() # Call the instance
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # Remember that review is a list. Only add words that not are in the stopword list
  review = ' '.join(review) # Make a long string instead of a list
  corpus.append(review)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)