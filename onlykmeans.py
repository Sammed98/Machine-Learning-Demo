import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def KMEANS(num_blobs,num_datapoints, num_clusters, random_seed):
    X, y = make_blobs(random_state = random_seed, n_samples=num_datapoints, centers = num_blobs)
    rng = np.random.RandomState(random_seed)
    transformation = rng.normal(size=(2, 2))
    X = np.dot(X, transformation)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    return kmeans,X,y_pred
