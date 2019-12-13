import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode

from ml_ch02.ml_ch02_util import Perceptron
from ml_ch02.ml_ch02_util import plot_decision_regions


df = pd.read_csv('data/ml_ch02/iris.data', header=None)
y = df.iloc[0:100, 4].values
    
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

    #
    # 1
    #
    
plt.scatter(X[0:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
    
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url1 = b64encode(img.getvalue()).decode('ascii')
    
    #
    # 2
    #
    
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url2 = b64encode(img.getvalue()).decode('ascii')

    #
    # 3
    #

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)

ml02_plot_url3 = b64encode(img.getvalue()).decode('ascii')
