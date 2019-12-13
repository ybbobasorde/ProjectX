from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

df = pd.read_csv('data/ml_ch06/wdbc.data', header=None)
df.head()
df.shape

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

le.transform(['M', 'B'])

X_train, X_test, y_train, y_test =     train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)


# ## Combining transformers and estimators in a pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs', multi_class='auto'))

pipe_lr.fit(X_train, y_train)


# # Using k-fold cross validation to assess model performance

# ## K-fold cross-validation

kfold = StratifiedKFold(n_splits=10,
                        random_state=1).split(X_train, y_train)

ml06_url2 = []

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    ml06_url2.insert(k, 'Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
ml06_url3 = 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))


scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
ml06_url4 = 'CV accuracy scores: %s' % scores
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
ml06_url5 = 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))

