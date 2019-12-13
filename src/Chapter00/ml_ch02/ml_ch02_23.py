import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode

from ml_ch02.ml_ch02_util import AdalineGD

    
df = pd.read_csv('data/ml_ch02/iris.data', header=None)
y = df.iloc[0:100, 4].values
    
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

    #
    #
    #
    
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    
    
ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
    
    #
    # 6
    #
    
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
    
plt.tight_layout()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url6 = b64encode(img.getvalue()).decode('ascii')
