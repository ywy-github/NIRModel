import numpy as np
import pandas as pd
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


if __name__ == '__main__':
    # Load data from Excel file
    data = pd.read_excel("../excels/CLAHE/二期.xlsx")

    # Extract relevant columns
    pred = data["pred"]
    prob = data["prob"]
    label = data["label"]

    # Convert to NumPy arrays
    test_pred = np.array(pred)  # Convert list to NumPy array
    test_targets = np.array(label)

    # Calculate metrics
    test_acc, test_sen, test_spe = all_metrics(label, prob)  # Ensure all_metrics function is defined

    # Calculate AUC
    test_auc = roc_auc_score(test_targets, test_pred)

    # Print results
    print("测试集 acc: {:.4f}".format(test_acc) + " sen: {:.4f}".format(test_sen) +
          " spe: {:.4f}".format(test_spe) + " auc: {:.4f}".format(test_auc))