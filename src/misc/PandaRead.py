""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# PandaRead.py:  Read table with pandas and use kmeans to find clusters
 
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
parts = pd.read_table("C:\ElemnPart.dat", delim_whitespace = True)
data = parts.drop("Name", axis=1)                        # Drop this column
data.head()
X = np.array(data["Number"])                                   # 1st column
y = np.array(data['Mass'])                                     # 2nd column
kmeans = KMeans(n_clusters = 3, random_state = 42)   # Random init clusters
kmeans.fit(data)                                        # Computes clusters
kmeans.predict(data)                              # Predict closest cluster
kmeans.labels_   
cc = kmeans.cluster_centers_                                    # Centroids
print(cc)                                                  # Show centroids
fig, ax = plt.subplots()
plt.xlabel("N")
plt.ylabel("Code")
plt.scatter(X[:],y[:],c=kmeans.labels_, marker="^")               # Arrows
plt.scatter(cc[:,0],cc[:,1],c='red', marker="D")                # Diamonds
plt.show()