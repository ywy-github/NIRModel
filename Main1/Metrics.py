import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


def all_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sen = np.round(sensitivity_score(y_true, y_pred), 4)
    spe = np.round(specificity_score(y_true, y_pred), 4)
    acc = np.round(accuracy_score(y_true, y_pred), 4)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]

    return acc,sen,spe