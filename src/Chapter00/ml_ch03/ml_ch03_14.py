import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from distutils.version import LooseVersion
from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    # 6
    #
    
#lr.predict_proba(X_test_std[:3, :])
#lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
#lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
#lr.predict(X_test_std[:3, :])
#lr.predict(X_test_std[0, :].reshape(1, -1))

weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1, solver='lbfgs', multi_class='auto')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml03_plot_url6 = b64encode(img.getvalue()).decode('ascii')

