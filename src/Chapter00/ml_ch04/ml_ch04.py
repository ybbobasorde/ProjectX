
# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 4 - Building Good Training Sets – Data Preprocessing

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# In[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Overview

# - [Dealing with missing data](#Dealing-with-missing-data)
#   - [Identifying missing values in tabular data](#Identifying-missing-values-in-tabular-data)
#   - [Eliminating samples or features with missing values](#Eliminating-samples-or-features-with-missing-values)
#   - [Imputing missing values](#Imputing-missing-values)
#   - [Understanding the scikit-learn estimator API](#Understanding-the-scikit-learn-estimator-API)
# - [Handling categorical data](#Handling-categorical-data)
#   - [Nominal and ordinal features](#Nominal-and-ordinal-features)
#   - [Mapping ordinal features](#Mapping-ordinal-features)
#   - [Encoding class labels](#Encoding-class-labels)
#   - [Performing one-hot encoding on nominal features](#Performing-one-hot-encoding-on-nominal-features)
# - [Partitioning a dataset into a separate training and test set](#Partitioning-a-dataset-into-seperate-training-and-test-sets)
# - [Bringing features onto the same scale](#Bringing-features-onto-the-same-scale)
# - [Selecting meaningful features](#Selecting-meaningful-features)
#   - [L1 and L2 regularization as penalties against model complexity](#L1-and-L2-regularization-as-penalties-against-model-omplexity)
#   - [A geometric interpretation of L2 regularization](#A-geometric-interpretation-of-L2-regularization)
#   - [Sparse solutions with L1 regularization](#Sparse-solutions-with-L1-regularization)
#   - [Sequential feature selection algorithms](#Sequential-feature-selection-algorithms)
# - [Assessing feature importance with Random Forests](#Assessing-feature-importance-with-Random-Forests)
# - [Summary](#Summary)


# In[2]:

from io import StringIO
from io import BytesIO
from base64 import b64encode
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ml_ch04.ml_ch04_util import SBS


# # Dealing with missing data

# ## Identifying missing values in tabular data

# In[3]:



csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''


# If you are using Python 2.7, you need
# to convert the string to unicode:

df = pd.read_csv(StringIO(csv_data))
df

# In[4]:

df.isnull().sum()

# In[5]:


# access the underlying NumPy array
# via the `values` attribute
df.values


# ## Eliminating samples or features with missing values

# In[6]:


# remove rows that contain missing values

df.dropna(axis=0)

# In[7]:


# remove columns that contain missing values

df.dropna(axis=1)

# In[8]:


# remove columns that contain missing values

df.dropna(axis=1)

# In[9]:


# only drop rows where all columns are NaN

df.dropna(how='all')  

# In[10]:


# drop rows that have less than 3 real values 

df.dropna(thresh=4)

# In[11]:


# only drop rows where NaN appear in specific columns (here: 'C')

df.dropna(subset=['C'])

# ## Imputing missing values

# In[12]:


# again: our original array
df.values

# In[13]:


# impute missing values via the column mean

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

# ## Understanding the scikit-learn estimator API

# In[14]:




# In[15]:





# # Handling categorical data

# ## Nominal and ordinal features

# In[16]:

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
df

# ## Mapping ordinal features

# In[17]:


size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df


# In[18]:


inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# ## Encoding class labels

# In[19]:

# create a mapping dict
# to convert class labels from strings to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping

# In[20]:


# to convert class labels from strings to integers
df['classlabel'] = df['classlabel'].map(class_mapping)
df

# In[21]:


# reverse the class label mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df


# In[22]:

# Label encoding with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

# In[23]:

# reverse mapping
class_le.inverse_transform(y)

# ## Performing one-hot encoding on nominal features

# In[24]:

X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X


# In[25]:

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

# In[26]:


# return dense array so that we can skip
# the toarray step

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
ohe.fit_transform(X)

# In[27]:


# one-hot encoding via pandas

pd.get_dummies(df[['price', 'color', 'size']])

# In[28]:


# multicollinearity guard in get_dummies

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)

# In[29]:


# multicollinearity guard for the OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()[:, 1:]

# # Partitioning a dataset into a seperate training and test set

# In[30]:


#df_wine = pd.read_csv('https://archive.ics.uci.edu/'
#                      'ml/machine-learning-databases/wine/wine.data',
#                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

df_wine = pd.read_csv('data/ml_ch04/wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
ml04_url1 = np.unique(df_wine['Class label'])
df_wine.head()

# In[31]:

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

# # Bringing features onto the same scale

# In[32]:

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# In[33]:

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# A visual example:

# In[34]:

ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())
ml04_url2 = (ex - ex.mean()) / ex.std()

# Please note that pandas uses ddof=1 (sample standard deviation) 
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))
ml04_url3 = (ex - ex.min()) / (ex.max() - ex.min())

# # Selecting meaningful features

# ...

# ## L1 and L2 regularization as penalties against model complexity

# ## A geometric interpretation of L2 regularization

# In[35]:




# In[36]:




# ## Sparse solutions with L1-regularization

# In[37]:




# For regularized models in scikit-learn that support L1 regularization, we can simply set the `penalty` parameter to `'l1'` to obtain a sparse solution:

# In[38]:

LogisticRegression(penalty='l1', solver='lbfgs', multi_class='auto')

# Applied to the standardized Wine data ...

# In[39]:

lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', multi_class='auto')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
ml04_url4 = lr.score(X_train_std, y_train)
print('Test accuracy:', lr.score(X_test_std, y_test))
ml04_url5 = lr.score(X_test_std, y_test)

# In[40]:

lr.intercept_

# In[41]:

np.set_printoptions(8)

# In[42]:

#lr.coef_[lr.coef_!=0].shape

# In[43]:

lr.coef_

# In[44]:


fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l2', C=10.**c, random_state=0, solver='lbfgs', multi_class='auto')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('images/04_07.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml04_plot_url1 = b64encode(img.getvalue()).decode('ascii')

# ## Sequential feature selection algorithms

# In[45]:

    #
    #
    #
 

# In[46]:

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml04_plot_url2 = b64encode(img.getvalue()).decode('ascii')

# In[47]:

    #
    #
    #

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])
ml04_url6 = df_wine.columns[1:][k3]

# In[48]:

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
ml04_url7 = knn.score(X_train_std, y_train)
print('Test accuracy:', knn.score(X_test_std, y_test))
ml04_url8 = knn.score(X_test_std, y_test)

# In[49]:

knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
ml04_url9 = knn.score(X_train_std[:, k3], y_train)
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
ml04_url10 = knn.score(X_test_std[:, k3], y_test)


# # Assessing feature importance with Random Forests

# In[50]:


feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

ml04_url11 = []
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    ml04_url11.insert(f, "%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml04_plot_url3 = b64encode(img.getvalue()).decode('ascii')

    #
    #
    #

# In[51]:

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of samples that meet this criterion:', 
      X_selected.shape[0])
ml04_url12 = X_selected.shape[0]

# Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):

# In[52]:


ml04_url13 = []
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    ml04_url13.insert(f, "%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

# # Summary

# ...
