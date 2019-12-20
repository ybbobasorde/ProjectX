#
# Predictive analytics with ensemble learning: Grid Search
#
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load input data
input_file = 'data/ai_ch03/data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate input data into three classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Split the data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Define the parameter grid 
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                   {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
                 ]

metrics = ['precision_weighted', 'recall_weighted']

ai03_4_url1 = []
ai03_4_url1_ctr=0
for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)
    ai03_4_url1_str = "##### Searching optimal parameters for", metric
    ai03_4_url1.insert(ai03_4_url1_ctr, ai03_4_url1_str)
    ai03_4_url1_ctr = ai03_4_url1_ctr + 1

    classifier = GridSearchCV(
            ExtraTreesClassifier(random_state=0), 
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    ai03_4_url2 = "\nGrid scores for the parameter grid:"
    ai03_4_url11 = []
    ai03_4_url11_ctr = 0
    for avg_score, _ in classifier.cv_results_['params']:        
        print(avg_score, '-->', classifier.cv_results_['params'])
        ai03_4_url11_str = avg_score, '-->', classifier.cv_results_['params']
        ai03_4_url11.insert(ai03_4_url11_ctr, ai03_4_url11_str)
        ai03_4_url11_ctr = ai03_4_url11_ctr + 1

    print("\nBest parameters:", classifier.best_params_)
    ai03_4_url3 = "Best parameters:", classifier.best_params_

    y_pred = classifier.predict(X_test)
    print("\nPerformance report:\n")
    ai03_4_url4 = "\nPerformance report:\n"
    print(classification_report(y_test, y_pred))
    ai03_4_url5 = classification_report(y_test, y_pred)

