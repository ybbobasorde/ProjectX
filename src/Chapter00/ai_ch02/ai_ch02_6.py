from io import BytesIO
from base64 import b64encode

from ai_ch02.ai_ch02_util import visualize_classifier

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from sklearn import linear_model 
#from utilities import visualize_classifier

# Define sample input data
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
#classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

# Train the classifier
classifier.fit(X, y)

# Visualize the performance of the classifier 
plt = visualize_classifier(classifier, X, y)

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai02_plot_url4 = b64encode(img.getvalue()).decode('ascii')