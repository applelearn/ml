import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

# Generate dataset
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster centroids are:\n", centroids)

# New data point
pred, _ = make_blobs(n_samples=1, centers=1, n_features=2, random_state=99)
print("New data point:", pred)
print("New data point belongs to cluster:", kmeans.predict(pred)[0])

# Plotting
colors = ['red', 'green', 'blue']
centroid_markers = ['X', 'D', 'P']  # Different shapes for centroids
point_marker = 'o'  # Shape for regular data points
new_point_marker = '*'  # Shape for the new point

plt.figure(figsize=(8, 6))

# Plot each cluster with a unique color
for i in range(3):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], label=f'Cluster {i}', s=30, alpha=0.6)
    plt.scatter(centroids[i, 0], centroids[i, 1], 
                c='black', marker=centroid_markers[i], 
                s=200, edgecolor='white', label=f'Centroid {i}')

# Plot new data point
plt.scatter(pred[0][0], pred[0][1], 
            c='yellow', edgecolor='black', 
            marker=new_point_marker, s=200, label='New Point')

plt.title("KMeans Clustering with New Data Point")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()