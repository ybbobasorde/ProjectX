
# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 2 - Training Machine Learning Algorithms for Classification

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# In[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# ### Overview
# 

# - [Artificial neurons � a brief glimpse into the early history of machine learning](#Artificial-neurons-a-brief-glimpse-into-the-early-history-of-machine-learning)
#     - [The formal definition of an artificial neuron](#The-formal-definition-of-an-artificial-neuron)
#     - [The perceptron learning rule](#The-perceptron-learning-rule)
# - [Implementing a perceptron learning algorithm in Python](#Implementing-a-perceptron-learning-algorithm-in-Python)
#     - [An object-oriented perceptron API](#An-object-oriented-perceptron-API)
#     - [Training a perceptron model on the Iris dataset](#Training-a-perceptron-model-on-the-Iris-dataset)
# - [Adaptive linear neurons and the convergence of learning](#Adaptive-linear-neurons-and-the-convergence-of-learning)
#     - [Minimizing cost functions with gradient descent](#Minimizing-cost-functions-with-gradient-descent)
#     - [Implementing an Adaptive Linear Neuron in Python](#Implementing-an-Adaptive-Linear-Neuron-in-Python)
#     - [Improving gradient descent through feature scaling](#Improving-gradient-descent-through-feature-scaling)
#     - [Large scale machine learning and stochastic gradient descent](#Large-scale-machine-learning-and-stochastic-gradient-descent)
# - [Summary](#Summary)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode

from ml_ch02.ml_ch02_util import Perceptron
from ml_ch02.ml_ch02_util import AdalineGD
from ml_ch02.ml_ch02_util import AdalineSGD
from ml_ch02.ml_ch02_util import plot_decision_regions

# # Artificial neurons - a brief glimpse into the early history of machine learning

# In[3]:




# ## The formal definition of an artificial neuron

# In[4]:




# ## The perceptron learning rule

# In[5]:




# In[6]:





# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API

# In[7]:

# In[8]:


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))



# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading-in the Iris data

# In[9]:


#df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#        'machine-learning-databases/iris/iris.data', header=None)
#df.tail()


# <hr>
# 
# ### Note:
# 
# 
# You can find a copy of the Iris dataset (and all other datasets used in this book) in the code bundle of this book, which you can use if you are working offline or the UCI server at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data is temporarily unavailable. For instance, to load the Iris dataset from a local directory, you can replace the line 
# 
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#         'machine-learning-databases/iris/iris.data', header=None)
#  
# by
#  
#     df = pd.read_csv('your/local/path/to/iris.data', header=None)
# 

# In[10]:


df = pd.read_csv('data/ml_ch02/iris.data', header=None)
df.tail()

# <hr>


# ### Plotting the Iris data

# In[11]:

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

    #
    # 1
    #

# plot data
plt.scatter(X[0:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
    
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
# plt.savefig('images/02_06.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url1 = b64encode(img.getvalue()).decode('ascii')
    
    #
    # 2
    #


# ### Training the perceptron model

# In[12]:


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
# plt.savefig('images/02_07.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url2 = b64encode(img.getvalue()).decode('ascii')

    #
    # 3
    #


# ### A function for plotting decision regions

# In[13]:


# In[14]:


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
# plt.savefig('images/02_08.png', dpi=300)
#plt.show()
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)

ml02_plot_url3 = b64encode(img.getvalue()).decode('ascii')

    #
    # 4
    #

# # Adaptive linear neurons and the convergence of learning

# ...

# ## Minimizing cost functions with gradient descent

# In[15]:




# In[16]:





# ## Implementing an adaptive linear neuron in Python

# In[17]:


# In[18]:

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

# plt.savefig('images/02_11.png', dpi=300)
#plt.show()
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)

ml02_plot_url4 = b64encode(img.getvalue()).decode('ascii')

    #
    # 5
    #

# In[19]:





# ## Improving gradient descent through feature scaling

# In[20]:




# In[21]:


# standardize features    
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    

# In[22]:

   
ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
    
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/02_14_1.png', dpi=300)
#plt.show()
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)

ml02_plot_url5 = b64encode(img.getvalue()).decode('ascii')
    
    #
    # 6
    #
    
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
    
plt.tight_layout()
# plt.savefig('images/02_14_2.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url6 = b64encode(img.getvalue()).decode('ascii')

    #
    # 7
    #
 

# ## Large scale machine learning and stochastic gradient descent

# In[23]:

# In[24]:

   
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
    
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
    
plt.tight_layout()
# plt.savefig('images/02_15_1.png', dpi=300)
#plt.show()
    
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url7 = b64encode(img.getvalue()).decode('ascii')
    
    #
    # 8
    #
    
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
    
plt.tight_layout()
# plt.savefig('images/02_15_2.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml02_plot_url8 = b64encode(img.getvalue()).decode('ascii')


# In[25]:


ada.partial_fit(X_std[0, :], y[0])



# # Summary

# ...

# --- 
# 
# Readers may ignore the following cell

# In[9]:




# In[ ]:

