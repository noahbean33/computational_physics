""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# KmeansCluster.py:  Clustering with sklearn's KMeans

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
X = np.array([
[1, 0], [2, 0.511], [3, 105.65], [4, 105.6583], [5, 134.98],
[6, 139.57],[7,139.57],[8,547.86],[9,497.68],[10,493.677],
[11,938.2721],[12,939.5654],[13,1115.68],[14,1180.37],
[15,1197.5],[16,1314.86], [17,1321.71],[18,1672.45]
])
kmeans = KMeans(n_clusters=3, random_state=42)       # 3 random centroids
kmeans.fit(X)                                       # Compute  clustering
kmeans.predict(X)                               # Predict closest cluster  
kmeans.labels_   
cc = kmeans.cluster_centers_                            # Cluster centers	
print("cc:",cc) 	
fig, ax = plt.subplots()
plt.xlabel("N")
plt.ylabel("Code")
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_, marker="^")
plt.scatter(cc[:,0],cc[:,1],c='red', marker="D") #with diamonds
plt.show()