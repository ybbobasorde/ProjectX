#
# Building recommender systems: K-Nearest Neighbors
#
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from sklearn.neighbors import NearestNeighbors

# Input data
X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], 
        [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9],
        [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

# Number of nearest neighbors
k = 5

# Test datapoint 
test_datapoint = [4.3, 2.7]

# Plot input data 
plt.figure()
plt.title('Input data')
plt.scatter(X[:,0], X[:,1], marker='o', s=75, color='black')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai05_3_plot_url1 =  b64encode(img.getvalue()).decode('ascii')

# Build K Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
distances, indices = knn_model.kneighbors([test_datapoint])

# Print the 'k' nearest neighbors
print("\nK Nearest Neighbors:")
ai05_3_url1 = []
ctr = 0
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank) + " ==>", X[index])
    ai05_3_url1_str = str(rank) + " ==>", X[index]
    ai05_3_url1.insert(ctr, ai05_3_url1_str)
    ctr = ctr + 1

# Visualize the nearest neighbors along with the test datapoint 
plt.figure()
plt.title('Nearest neighbors')
plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='k')
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1], 
        marker='o', s=250, color='k', facecolors='none')
plt.scatter(test_datapoint[0], test_datapoint[1],
        marker='x', s=75, color='k')

#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai05_3_plot_url2 =  b64encode(img.getvalue()).decode('ascii')
