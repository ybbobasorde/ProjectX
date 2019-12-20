#
# Building recommender systems: Nearest Neighbor Classifier
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
from base64 import b64encode
from sklearn import neighbors

# Load input data
input_file = 'data/ai_ch05/data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(np.int)

# Plot input data
plt.figure()
plt.title('Input data')
marker_shapes = 'v^os'
mapper = [marker_shapes[i] for i in y]
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], 
            s=75, edgecolors='black', facecolors='none')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai05_5_plot_url1 =  b64encode(img.getvalue()).decode('ascii')

# Number of nearest neighbors 
num_neighbors = 12

# Step size of the visualization grid
step_size = 0.01  

# Create a K Nearest Neighbours classifier model 
classifier = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')

# Train the K Nearest Neighbours model
classifier.fit(X, y)

# Create the mesh to plot the boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), 
        np.arange(y_min, y_max, step_size))

# Evaluate the classifier on all the points on the grid 
output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

# Visualize the predicted output 
output = output.reshape(x_values.shape)
plt.figure()
plt.pcolormesh(x_values, y_values, output, cmap=cm.Paired)

# Overlay the training points on the map
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], 
            s=50, edgecolors='black', facecolors='none')

plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())
plt.title('K Nearest Neighbors classifier model boundaries')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai05_5_plot_url2 =  b64encode(img.getvalue()).decode('ascii')

# Test input datapoint
test_datapoint = [5.1, 3.6]
plt.figure()
plt.title('Test datapoint')
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], 
            s=75, edgecolors='black', facecolors='none')

plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', 
        linewidth=6, s=200, facecolors='black')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai05_5_plot_url3 =  b64encode(img.getvalue()).decode('ascii')

# Extract the K nearest neighbors
_, indices = classifier.kneighbors([test_datapoint])
indices = indices.astype(np.int)[0]

# Plot k nearest neighbors
plt.figure()
plt.title('K Nearest Neighbors')

for i in indices:
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[y[i]], 
            linewidth=3, s=100, facecolors='black')

plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', 
        linewidth=6, s=200, facecolors='black')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai05_5_plot_url4 =  b64encode(img.getvalue()).decode('ascii')


for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], 
            s=75, edgecolors='black', facecolors='none')

print("Predicted output:", classifier.predict([test_datapoint])[0])
ai05_5_url1 = classifier.predict([test_datapoint])[0]

#plt.show()

