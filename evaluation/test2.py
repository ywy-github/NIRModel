import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

if __name__ == '__main__':
    # Define cutoff values
    cutoffs = np.linspace(0, 1, 21)

    # Dummy values for the trends based on provided average values
    acc = [0.73 for _ in cutoffs]
    sen = [0.54 for _ in cutoffs]
    spe = [0.82 for _ in cutoffs]
    auc = [0.76 for _ in cutoffs]

    # Plotting 一期数据集
    plt.figure(figsize=(10, 6))
    plt.plot(cutoffs, acc, label='ACC', color='blue')
    plt.plot(cutoffs, sen, label='SEN', color='orange')
    plt.plot(cutoffs, spe, label='SPE', color='green')
    plt.plot(cutoffs, auc, label='AUC', color='red')

    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Cutoff')
    plt.ylabel('Value')
    plt.title('Metrics with Cutoff for 一期数据集')
    plt.legend()
    plt.show()
