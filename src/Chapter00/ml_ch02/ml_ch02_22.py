import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode

from ml_ch02.ml_ch02_util import AdalineGD
from ml_ch02.ml_ch02_util import plot_decision_regions

    
df = pd.read_csv('data/ml_ch02/iris.data', header=None)
y = df.iloc[0:100, 4].values
    
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

    #
    # 5
    #
    
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    
    
ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
    
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)

ml02_plot_url5 = b64encode(img.getvalue()).decode('ascii')
