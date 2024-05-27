import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

if __name__ == '__main__':

    # Read the Excel file
    file_path = '../document/excels/SRCNet/一期+二期.xlsx'  # Replace with the actual file path
    data = pd.read_excel(file_path)

    # Extract the prediction probabilities and true labels
    pred_probs = data['pred']
    true_labels = data['label']

    # Define the cutoff values (21 points from 0 to 1 with step 0.05)
    cutoffs = np.linspace(0, 1, 21)

    # Initialize lists to store the metrics
    accs = []
    sens = []
    spes = []
    aucs = []

    # Calculate the AUC once as it does not depend on the cutoff
    auc = roc_auc_score(true_labels, pred_probs)

    # Calculate the metrics at each cutoff value
    for cutoff in cutoffs:
        # Binarize the predictions based on the cutoff
        pred_labels = (pred_probs >= cutoff).astype(int)

        # Calculate the metrics
        acc = accuracy_score(true_labels, pred_labels)
        sen = recall_score(true_labels, pred_labels)  # Sensitivity is the same as recall
        spe = recall_score(true_labels, pred_labels, pos_label=0)  # Specificity

        # Append the metrics to the lists
        accs.append(acc)
        sens.append(sen)
        spes.append(spe)
        aucs.append(auc)  # AUC remains the same for all cutoffs

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(cutoffs, accs, label='ACC', color='blue')
    plt.plot(cutoffs, sens, label='SEN', color='orange')
    plt.plot(cutoffs, spes, label='SPE', color='green')
    plt.plot(cutoffs, aucs, label='AUC', color='red')

    # Set the x and y axis intervals
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    # Add labels and title
    plt.xlabel('Cutoff')
    plt.ylabel('Value')
    plt.title('Metrics with Cutoff')
    plt.legend()

    # Show the plot
    plt.show()
