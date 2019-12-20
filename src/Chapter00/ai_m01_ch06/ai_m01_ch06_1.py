#K-means clustering
#Build with Sklearn
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
import pandas as pd
from base64 import b64encode
from io import BytesIO
from sklearn.cluster import KMeans  
from matplotlib import pyplot as plt

#I.The training Dataset 
dataset = pd.read_csv('data/ai_m01_ch06/data.csv')
print (dataset.head())
ai06_1_url1 = dataset.head()
print(dataset)
ai06_1_url2 = dataset
'''Output of print(dataset)
      Distance  location
0           80        53
1           18         8
2           55        38
...
'''

#II.Hyperparameters
# Features = 2
k = 6
kmeans = KMeans(n_clusters=k)

#III.K-means clustering algorithm
kmeans = kmeans.fit(dataset)         #Computing k-means clustering
gcenters = kmeans.cluster_centers_   # the geometric centers or centroids
print("The geometric centers or centroids:")
print(gcenters)
ai06_1_url3 = gcenters
'''Ouput of centroid coordinates
[[ 48.7986755   85.76688742]
 [ 32.12590799  54.84866828]
 [ 96.06151645  84.57939914]
 [ 68.84578885  55.63226572]
 [ 48.44532803  24.4333996 ]
 [ 21.38965517  15.04597701]]
'''


#IV.Defining the Result labels 
labels = kmeans.labels_
colors = ['blue','red','green','black','yellow','brown','orange']


#V.Displaying the results : datapoints and clusters
y = 0
for x in labels:
    plt.scatter(dataset.iloc[y,0], dataset.iloc[y,1],color=colors[x])
    y+=1       
for x in range(k):
    lines = plt.plot(gcenters[x,0],gcenters[x,1],'kx')    

title = ('No of clusters (k) = {}').format(k)
plt.title(title)
plt.xlabel('Distance')
plt.ylabel('Location')
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai06_plot_url1 = b64encode(img.getvalue()).decode('ascii')

#VI.Test dataset and prediction
x_test = [[40.0,67],[20.0,61],[90.0,90],
          [50.0,54],[20.0,80],[90.0,60]]
prediction = kmeans.predict(x_test)
print("The predictions:")
print (prediction)
ai06_1_url4 = prediction
'''
Output of the cluster number of each example
[3 3 2 3 3 4]
'''
