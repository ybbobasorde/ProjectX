#
# Detecting patterns with unsupervised learning: Market Segmentation
#
import csv
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from sklearn.cluster import MeanShift, estimate_bandwidth

# Load data from input file
input_file = 'data/ai_ch04/sales.csv'
file_reader = csv.reader(open(input_file, 'r'), delimiter=',')
X = []
for count, row in enumerate(file_reader):
    if not count:
        names = row[1:]
        continue

    X.append([float(x) for x in row[1:]])

# Convert to numpy array
X = np.array(X)

# Estimating the bandwidth of input data
bandwidth = estimate_bandwidth(X, quantile=0.8, n_samples=len(X))

# Compute clustering with MeanShift
meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_model.fit(X)
labels = meanshift_model.labels_
cluster_centers = meanshift_model.cluster_centers_
num_clusters = len(np.unique(labels))

print("\nNumber of clusters in input data =", num_clusters)
ai04_4_url1 = num_clusters

print("\nCenters of clusters:")
ai04_4_url2 = "Centers of clusters:"
print('\t'.join([name[:3] for name in names]))
ai04_4_url3 = '\t'.join([name[:3] for name in names])
ai04_4_url4 = []
ctr = 0
for cluster_center in cluster_centers:
    print('\t'.join([str(int(x)) for x in cluster_center]))
    ai04_4_url4.insert(ctr, '\t'.join([str(int(x)) for x in cluster_center]))
    ctr = ctr + 1

# Extract two features for visualization 
cluster_centers_2d = cluster_centers[:, 1:3]

# Plot the cluster centers 
plt.figure()
plt.scatter(cluster_centers_2d[:,0], cluster_centers_2d[:,1], 
        s=120, edgecolors='black', facecolors='none')

offset = 0.25
plt.xlim(cluster_centers_2d[:,0].min() - offset * cluster_centers_2d[:,0].ptp(),
        cluster_centers_2d[:,0].max() + offset * cluster_centers_2d[:,0].ptp(),)
plt.ylim(cluster_centers_2d[:,1].min() - offset * cluster_centers_2d[:,1].ptp(),
        cluster_centers_2d[:,1].max() + offset * cluster_centers_2d[:,1].ptp())

plt.title('Centers of 2D clusters')
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai04_4_plot_url1 =  b64encode(img.getvalue()).decode('ascii')
