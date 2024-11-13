from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report
import numpy as np
y_true = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0, 2])
y_pred = np.array([0, 2, 2, 1, 0, 1, 2, 1, 0, 1])

print('accuray_score:', accuracy_score(y_true, y_pred))
print('recall_score:', recall_score(y_true, y_pred, average='weighted'))
print('precision_score:', precision_score(y_true, y_pred, average='weighted'))
print('f1_score:', f1_score(y_true, y_pred, average='weighted'))
print('confusion_matrix:', classification_report(y_true, y_pred))
