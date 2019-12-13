from base64 import b64encode
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('data/ml_ch10/housing.data.txt',
                  sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()



# # Implementing an ordinary least squares linear regression model

# ...

# ## Solving regression for regression parameters with gradient descent

# In[9]:


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# In[10]:


X = df[['RM']].values
y = df['MEDV'].values


# In[11]:




sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# In[12]:


lr = LinearRegressionGD()
lr.fit(X_std, y_std)


# In[13]:


plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
#plt.tight_layout()
#plt.savefig('images/10_05.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml10_plot_url3 = b64encode(img.getvalue()).decode('ascii')


# In[14]:


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 


# In[15]:


lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

#plt.savefig('images/10_06.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml10_plot_url4 = b64encode(img.getvalue()).decode('ascii')


# In[16]:


print('Slope: %.3f' % lr.w_[1])
ml10_url1 = 'Slope: %.3f' % lr.w_[1]
print('Intercept: %.3f' % lr.w_[0])
ml10_url2 = 'Intercept: %.3f' % lr.w_[0]

# In[17]:


num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))
ml10_url3 = "Price in $1000s: %.3f" % sc_y.inverse_transform(price_std)



# ## Estimating the coefficient of a regression model via scikit-learn

# In[18]:




# In[19]:


slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
ml10_url4 = 'Slope: %.3f' % slr.coef_[0]
print('Intercept: %.3f' % slr.intercept_)
ml10_url5 = 'Intercept: %.3f' % slr.intercept_


# In[20]:


lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

#plt.savefig('images/10_07.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml10_plot_url5 = b64encode(img.getvalue()).decode('ascii')



# **Normal Equations** alternative:

# In[21]:


# adding a column vector of "ones"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print('Slope: %.3f' % w[1])
ml10_url6 = 'Slope: %.3f' % w[1]
print('Intercept: %.3f' % w[0])
ml10_url7 = 'Intercept: %.3f' % w[0]


