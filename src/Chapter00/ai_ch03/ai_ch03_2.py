#
# Predictive analytics with ensemble learning: Decision Trees
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ai_utils import visualize_classifier

# Load input data
input_file = 'data/ai_ch03/data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate input data into two classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Visualize input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', 
        edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', 
        edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

# Split data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

# Decision Trees classifier 
params = {'random_state': 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)

ai03_2_plot_url1 = visualize_classifier(classifier, X_train, y_train, 'Training dataset')

y_test_pred = classifier.predict(X_test)
ai03_2_plot_url2 = visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
ai03_2_url1 = "\n" + "#"*40
print("\nClassifier performance on training dataset\n")
ai03_2_url2 = "\nClassifier performance on training dataset\n"
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
ai03_2_url3 = classification_report(y_train, classifier.predict(X_train), target_names=class_names)
print("#"*40 + "\n")
ai03_2_url4 = "#"*40 + "\n"

print("#"*40)
ai03_2_url5 = "#"*40
print("\nClassifier performance on test dataset\n")
ai03_2_url6 = "\nClassifier performance on test dataset\n"
print(classification_report(y_test, y_test_pred, target_names=class_names))
ai03_2_url7 = classification_report(y_test, y_test_pred, target_names=class_names)
print("#"*40 + "\n")
ai03_2_url8 = "#"*40 + "\n"

#plt.show()
