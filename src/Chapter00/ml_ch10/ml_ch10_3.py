from base64 import b64encode
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('data/ml_ch10/housing.data.txt',
                  sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


X = df[['RM']].values
y = df['MEDV'].values



# # Fitting a robust regression model using RANSAC

# In[22]:



ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)


ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

#plt.savefig('images/10_08.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml10_plot_url6 = b64encode(img.getvalue()).decode('ascii')


# In[23]:


print('Slope: %.3f' % ransac.estimator_.coef_[0])
ml10_url8 = 'Slope: %.3f' % ransac.estimator_.coef_[0]
print('Intercept: %.3f' % ransac.estimator_.intercept_)
ml10_url9 = 'Intercept: %.3f' % ransac.estimator_.intercept_

