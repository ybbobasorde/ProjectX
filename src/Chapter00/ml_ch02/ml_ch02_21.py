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
    # 4
    #
    
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
    
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)

ml02_plot_url4 = b64encode(img.getvalue()).decode('ascii')

