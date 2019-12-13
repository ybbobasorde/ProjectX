from base64 import b64encode
from io import BytesIO
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/ml_ch06/wdbc.data', header=None)
df.head()
df.shape

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test =     train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)


# # Fine-tuning machine learning models via grid search


# ## Tuning hyperparameters via grid search 

# In[17]:




pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))



# # Looking at different performance evaluation metrics

# ...

# ## Reading a confusion matrix

# In[22]:




# In[23]:



pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# In[24]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ml06_plot_url3 = b64encode(img.getvalue()).decode('ascii')



# ### Additional Note

# Remember that we previously encoded the class labels so that *malignant* samples are the "postive" class (1), and *benign* samples are the "negative" class (0):

# In[25]:


le.transform(['M', 'B'])


# In[26]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# Next, we printed the confusion matrix like so:

# In[27]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# Note that the (true) class 0 samples that are correctly predicted as class 0 (true negatives) are now in the upper left corner of the matrix (index 0, 0). In order to change the ordering so that the true negatives are in the lower right corner (index 1,1) and the true positves are in the upper left, we can use the `labels` argument like shown below:

# In[28]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
print(confmat)


# We conclude:
# 
# Assuming that class 1 (malignant) is the positive class in this example, our model correctly classified 71 of the samples that belong to class 0 (true negatives) and 40 samples that belong to class 1 (true positives), respectively. However, our model also incorrectly misclassified 1 sample from class 0 as class 1 (false positive), and it predicted that 2 samples are benign although it is a malignant tumor (false negatives).


# ## Optimizing the precision and recall of a classification model

# In[29]:



print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
ml06_url11 = 'Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred)
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
ml06_url12 = 'Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred)
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
ml06_url13 = 'F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred)


# In[30]:



scorer = make_scorer(f1_score, pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
ml06_url14 = gs.best_score_
print(gs.best_params_)
ml06_url15 = gs.best_params_

