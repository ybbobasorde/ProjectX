#
# Classification and regression using supervised learning : Regressor Singlevar
#
import pickle
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from sklearn import linear_model

# Input file containing data
input_file = 'data/ai_ch02/data_singlevar_regr.txt' 

# Read data
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

# Create linear regressor object
regressor = linear_model.LinearRegression()

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = regressor.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
#plt.show()

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai02_plot_url1 = b64encode(img.getvalue()).decode('ascii')

# Compute performance metrics
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
ai02_9_url1 = round(sm.mean_absolute_error(y_test, y_test_pred), 2)
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
ai02_9_url2 = round(sm.mean_squared_error(y_test, y_test_pred), 2)
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
ai02_9_url3 = round(sm.median_absolute_error(y_test, y_test_pred), 2)
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
ai02_9_url4 = round(sm.explained_variance_score(y_test, y_test_pred), 2)
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
ai02_9_url5 = round(sm.r2_score(y_test, y_test_pred), 2)

# Model persistence
output_model_file = 'data/ai_ch02/model.pkl'

# Save the model
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Load the model
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Perform prediction on test data
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
ai02_9_url6 = round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)
