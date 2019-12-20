#
# Detecting patterns with unsupervised learning: Mean Shift
#
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from sklearn.cluster import MeanShift, estimate_bandwidth

# Load data from input file
X = np.loadtxt('data/ai_ch04/data_clustering.txt', delimiter=',')

# Estimate the bandwidth of X
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Cluster data with MeanShift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Extract the centers of clusters
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)
ai04_5_url1 = cluster_centers

# Estimate the number of clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)
ai04_5_url2 = num_clusters

# Plot the points and cluster centers
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # Plot points that belong to the current cluster
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')

    # Plot the cluster center
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', 
            markerfacecolor='black', markeredgecolor='black', 
            markersize=15)

plt.title('Clusters')
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai04_5_plot_url1 =  b64encode(img.getvalue()).decode('ascii')
