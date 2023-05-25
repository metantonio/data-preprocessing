# K-Means Clustering (works with N-Dimensions)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
col_start = 1
col_end = 5
col_range = col_end - col_start
X = dataset.iloc[
    :, col_start:col_end
].values  # all columns are features for clustering, i can forget some columns if i believe doesn't have impact, that's why i will keep annual income and spending score for teaching reasons. Range do not include las column
print("\nX: ", X) # X = [[1,2,3,...], [1,2,3....]]
# If i keeped 2 columns, i'll have a 2D cluster

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
def one_hot_encoder(array2D): # With featuring scale
    loop = 0
    for value in array2D[0]:
        print(value, "loop: ", loop)
        if isinstance(value, str):
            print("is string")
            ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [loop])], remainder="passthrough")  # remember change the index of column if there are a categorical data
            X_temp = np.array(
                ct.fit_transform(array2D)
            )  # on multiple linear regression we don't need apply scaling features
            print("transformed matrix of features: \n", X_temp)

            # Feature Scaling (in this model we don't split datase into test and training, bacause we want the correlation of all data)
            from sklearn.preprocessing import StandardScaler

            # features and result are in a very diferent scale and outside of [-3, 3] range, not dummy variables and as we are not in linear models, we need scale
            sc_X = (
                StandardScaler()
            )  # scaler that calculates mean and standard deviation of matrix of features
            sc_y = StandardScaler()
            X_temp = sc_X.fit_transform(X_temp)
        loop = loop + 1
    return X_temp
        
X = one_hot_encoder(X)
print("X after scaling:\n", X)


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)
print("X_kmeans: ",X, " , length: ", len(X))
print("y_kmeans: ",y_kmeans, " , length:",len(y_kmeans))

# Testing data:
X_test = [['Male', 19, 15, 39]]

# Visualising the clusters
if col_range == 2:
    plt.scatter(
        X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 1"
    )
    plt.scatter(
        X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Cluster 2"
    )
    plt.scatter(
        X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Cluster 3"
    )
    plt.scatter(
        X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="cyan", label="Cluster 4"
    )
    plt.scatter(
        X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="magenta", label="Cluster 5"
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c="yellow",
        label="Centroids",
    )
    plt.title("Clusters of customers")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()
