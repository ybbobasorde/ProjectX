import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from distutils.version import LooseVersion
from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_ch03.ml_ch03_util import plot_decision_regions

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    #
    #
    #

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

    #
    # 14
    #

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=1)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml03_plot_url14 = b64encode(img.getvalue()).decode('ascii')

