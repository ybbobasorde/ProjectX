import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version

from ml_ch03.ml_ch03_util import gini
from ml_ch03.ml_ch03_util import entropy
from ml_ch03.ml_ch03_util import error

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')

    #
    # 12
    #

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml03_plot_url12 = b64encode(img.getvalue()).decode('ascii')

