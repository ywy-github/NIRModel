import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, \
    recall_score, f1_score


def ROC_Curve(y_true,y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    print('AUC:{}'.format(roc_auc))
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', linewidth=2,
             label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def PR_Curve(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')  # 显示P-R曲线
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')  # 将曲线下部分面积填充
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall curve')
    plt.show()


# y_true:真实标签
# y_prob:预测标签
def plot_confusion_matrix(y_true, y_prob):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_prob)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 将混淆矩阵中的每个元素除以各自行的总和，以得到准确率

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ["benign", "malignant"]  # 类别标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在格子中显示准确率
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.4f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()


def compute(y_true,y_prob,y_pred):
    # ACC
    acc = accuracy_score(y_true, y_prob)

    # AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision
    precision = precision_score(y_true, y_prob)

    # Recall
    recall = recall_score(y_true, y_prob)

    # F1-score
    f1 = f1_score(y_true, y_prob)

    print("Accuracy: %.4f" % acc)
    print(("roc_auc: %.4f") % roc_auc)
    print("Precision: %.4f" % precision)
    print("Recall: %.4f" % recall)
    print("F1-score: %.4f" % f1)


if __name__ == '__main__':
    data = pd.read_excel("../models1/result/resnet18+删一个layer.xlsx")
    y_true = data.loc[:,"label"]
    y_prob = data.loc[:,"prob"]
    y_pred = data.loc[:,"pred"]
    # plot_confusion_matrix(y_true,y_prob)
    #
    # PR_Curve(y_true,y_pred)
    # ROC_Curve(y_true,y_pred)

    compute(y_true,y_prob,y_pred)