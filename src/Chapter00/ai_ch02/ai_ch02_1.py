#
# Classification and regression using supervised learning : Confusion Matrix
#
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define sample labels
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Create confusion matrix
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai02_plot_url1 = b64encode(img.getvalue()).decode('ascii')

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))
ai02_1_url1 = classification_report(true_labels, pred_labels, target_names=targets)