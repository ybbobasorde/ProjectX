#
# Detecting patterns with unsupervised learning: Clustering Quality
#
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from sklearn import metrics
from sklearn.cluster import KMeans

# Load data from input file
X = np.loadtxt('data/ai_ch04/data_quality.txt', delimiter=',')

# Plot input data
plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Initialize variables
scores = []
values = np.arange(2, 10)

# Iterate through the defined range
ai04_1_url1 = []
ai04_1_urls_ctr=0
for num_clusters in values:
    # Train the KMeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_, 
                metric='euclidean', sample_size=len(X))

    print("\nNumber of clusters =", num_clusters)
    ai04_1_url2_str = "Number of clusters =", num_clusters
    
    print("Silhouette score =", score)
    ai04_1_url3_str = "Silhouette score =", score
    
    ai04_1_url1_str = (ai04_1_url2_str, ai04_1_url3_str)
    ai04_1_url1.insert(ai04_1_urls_ctr, ai04_1_url1_str)
    ai04_1_urls_ctr = ai04_1_urls_ctr + 1

    scores.append(score)

# Plot silhouette scores
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs number of clusters')

# Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)
ai04_1_url2 = 'Optimal number of clusters =', num_clusters

#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
        
ai04_1_plot_url1 =  b64encode(img.getvalue()).decode('ascii')