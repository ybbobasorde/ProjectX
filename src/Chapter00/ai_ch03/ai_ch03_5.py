#
# Predictive analytics with ensemble learning: Random Forests
#
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
from ai_utils import visualize_classifier

# Argument parser 
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify data using \
            Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest='classifier_type', 
            required=True, choices=['rf', 'erf'], help="Type of classifier \
                    to use; can be either 'rf' or 'erf'")
    return parser

    #if __name__=='__main__':
    # Parse the input arguments
    #args = build_arg_parser().parse_args()
    #classifier_type = args.classifier_type
    
classifier_type = 'rf'
#classifier_type = 'erf'

# Load input data
input_file = 'data/ai_ch03/data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
    
# Separate input data into three classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Visualize input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', 
                edgecolors='black', linewidth=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', 
                edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', 
                edgecolors='black', linewidth=1, marker='^')
plt.title('Input data')

# Split data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Ensemble Learning classifier
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if classifier_type == 'rf':
    classifier = RandomForestClassifier(**params)
else:
    classifier = ExtraTreesClassifier(**params)

classifier.fit(X_train, y_train)
ai03_5_plot_url1 = visualize_classifier(classifier, X_train, y_train, 'Training dataset')

y_test_pred = classifier.predict(X_test)
ai03_5_plot_url2 = visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#"*40)
ai03_5_url1 = "#"*40
print("\nClassifier performance on training dataset\n")
ai03_5_url2 = "Classifier performance on training dataset"
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
ai03_5_url3 = classification_report(y_train, classifier.predict(X_train), target_names=class_names)
print("#"*40 + "\n")

print("#"*40)
ai03_5_url4 = "#"*40
print("\nClassifier performance on test dataset\n")
ai03_5_url5 = "Classifier performance on test dataset"
print(classification_report(y_test, y_test_pred, target_names=class_names))
ai03_5_url6 = classification_report(y_test, y_test_pred, target_names=class_names)
print("#"*40 + "\n")
ai03_5_url7 = "#"*40 

# Compute confidence
test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

print("\nConfidence measure:")
ai03_5_url8 = "Confidence measure:" 
ai03_5_url9 = []
ctr = 0
for datapoint in test_datapoints:
    probabilities = classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    ai03_5_url9_str = 'Datapoint:', datapoint
    ai03_5_url9.insert(ctr, ai03_5_url9_str) 
    ctr = ctr + 1 
    print('\nDatapoint:', datapoint)
    ai03_5_url9_str = 'Predicted class:', predicted_class
    ai03_5_url9.insert(ctr, ai03_5_url9_str) 
    ctr = ctr +1 
    print('Predicted class:', predicted_class)

# Visualize the datapoints
ai03_5_plot_url3 = visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints), 'Test datapoints')

#plt.show()

