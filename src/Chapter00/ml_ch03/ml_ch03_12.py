import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from distutils.version import LooseVersion
from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_ch03.ml_ch03_util import plot_decision_regions
from ml_ch03.ml_ch03_util import LogisticRegressionGD
from ml_ch03.ml_ch03_util import sigmoid
from ml_ch03.ml_ch03_util import cost_1
from ml_ch03.ml_ch03_util import cost_0

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

    #
    #
    #

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

    
    #
    # 3
    #

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml03_plot_url3 = b64encode(img.getvalue()).decode('ascii')

    #
    # 4
    #

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, max_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml03_plot_url4 = b64encode(img.getvalue()).decode('ascii')

