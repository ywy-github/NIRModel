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
    plt.plot(fpr, tpr, color='darkorange', linewidth=2,
             label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    # plt.title('Receiver Operating Characteristic Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()

def Multi_ROC_Curve():
    filepath = "../models2/excels"
    # data_AlexNet = pd.read_excel(filepath + "/AlexNet-5.xlsx")
    # data_DenseNet = pd.read_excel(filepath + "/DenseNet-13.xlsx")
    # data_MobileNet = pd.read_excel(filepath + "/MobileNet-13.xlsx")
    # data_Resnet18 = pd.read_excel(filepath + "/Resnet18-8.xlsx")
    # data_Resnet50 = pd.read_excel(filepath + "/AlexNet+CNN.xlsx")
    # data_RCNet = pd.read_excel(filepath + "/Vq-VAE-resnet18仅重构+分类器-71.xlsx")
    data_SRCNet = pd.read_excel(filepath + "/VQ-VAE-resnet18-29.xlsx")
    data_TSRCNet = pd.read_excel("../models2/筛查重构+分类_excels/筛查重构+分类-89.xlsx")

    # y_true_AlexNet = data_AlexNet.loc[:, "label"]
    # y_true_DenseNet = data_DenseNet.loc[:, "label"]
    # y_true_MobileNet = data_MobileNet.loc[:, "label"]
    # y_true_Resnet18 = data_Resnet18.loc[:, "label"]
    # y_true_Resnet50 = data_Resnet50.loc[:, "label"]
    # y_true_RCNet = data_RCNet.loc[:, "label"]
    y_true_SRCNet = data_SRCNet.loc[:, "label"]
    y_true_TSRCNet = data_TSRCNet.loc[:, "label"]

    # y_scores_AlexNet = data_AlexNet.loc[:, "pred"]
    # y_scores_DenseNet = data_DenseNet.loc[:, "pred"]
    # y_scores_MobileNet = data_MobileNet.loc[:, "pred"]
    # y_scores_Resnet18 = data_Resnet18.loc[:, "pred"]
    # y_scores_Resnet50 = data_Resnet50.loc[:, "pred"]
    # y_scores_RCNet = data_RCNet.loc[:, "pred"]
    y_scores_SRCNet = data_SRCNet.loc[:, "pred"]
    y_scores_TSRCNet = data_TSRCNet.loc[:, "pred"]

    # fpr_AlexNet, tpr_AlexNet, thresholds_AlexNet = roc_curve(y_true_AlexNet, y_scores_AlexNet)
    # fpr_DenseNet, tpr_DenseNet, thresholds_DenseNet = roc_curve(y_true_DenseNet,y_scores_DenseNet)
    # fpr_MobileNet, tpr_MobileNet, thresholds_MobileNet = roc_curve(y_true_MobileNet,y_scores_MobileNet)
    # fpr_Resnet18, tpr_Resnet18, thresholds_Resnet18 = roc_curve(y_true_Resnet18,y_scores_Resnet18)
    # fpr_Resnet50, tpr_Resnet50, thresholds_Resnet50 = roc_curve(y_true_Resnet50,y_scores_Resnet50)
    # fpr_RCNet, tpr_RCNet, thresholds_RCNet = roc_curve(y_true_RCNet, y_scores_RCNet)
    fpr_SRCNet, tpr_SRCNet, thresholds_SRCNet = roc_curve(y_true_SRCNet, y_scores_SRCNet)
    fpr_TSRCNet, tpr_TSRCNet, thresholds_TSRCNet = roc_curve(y_true_TSRCNet, y_scores_TSRCNet)

    # roc_auc_AlexNet = auc(fpr_AlexNet, tpr_AlexNet)
    # roc_auc_DenseNet = auc(fpr_DenseNet, tpr_DenseNet)
    # roc_auc_MobileNet = auc(fpr_MobileNet, tpr_MobileNet)
    # roc_auc_Resnet18 = auc(fpr_Resnet18, tpr_Resnet18)
    # roc_auc_Resnet50 = auc(fpr_Resnet50, tpr_Resnet50)
    # roc_auc_RCNet = auc(fpr_RCNet, tpr_RCNet)
    roc_auc_SRCNet = auc(fpr_SRCNet, tpr_SRCNet)
    roc_auc_TSRCNet = auc(fpr_TSRCNet, tpr_TSRCNet)

    plt.figure()

    # plt.plot(fpr_Resnet18, tpr_Resnet18, linewidth=2,
    #          label='Resnet18 (area = %0.4f)' % roc_auc_Resnet18)
    # plt.plot(fpr_Resnet50, tpr_Resnet50, linewidth=2,
    #          label='Resnet50 (area = %0.4f)' % roc_auc_Resnet50)
    # plt.plot(fpr_AlexNet, tpr_AlexNet, linewidth=2,
    #          label='AlexNet (area = %0.4f)' % roc_auc_AlexNet)
    # plt.plot(fpr_MobileNet, tpr_MobileNet, linewidth=2,
    #          label='MobileNet (area = %0.4f)' % roc_auc_MobileNet)
    # plt.plot(fpr_DenseNet, tpr_DenseNet, linewidth=2,
    #          label='DenseNet (area = %0.4f)' % roc_auc_DenseNet)
    # plt.plot(fpr_RCNet, tpr_RCNet, linewidth=2,
    #          label='RCNet (area = %0.4f)' % roc_auc_RCNet)
    plt.plot(fpr_SRCNet, tpr_SRCNet, linewidth=2,
             label='SRCNet (area = %0.4f)' % roc_auc_SRCNet)
    plt.plot(fpr_TSRCNet, tpr_TSRCNet, linewidth=2,
             label='TSRCNet (area = %0.4f)' % roc_auc_TSRCNet)

    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()


def PR_Curve(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')  # 显示P-R曲线
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')  # 将曲线下部分面积填充
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall', fontsize=14)
    plt.ylabel('precision', fontsize=14)
    # plt.title('precision-recall curve', fontsize=14)
    plt.show()

def Multi_PR_Curve():
    filepath = "../models2/excels"
    # data_AlexNet = pd.read_excel(filepath + "/AlexNet-5.xlsx")
    # data_DenseNet = pd.read_excel(filepath + "/DenseNet-13.xlsx")
    # data_MobileNet = pd.read_excel(filepath + "/MobileNet-13.xlsx")
    # data_Resnet18 = pd.read_excel(filepath + "/Resnet18-8.xlsx")
    # data_Resnet50 = pd.read_excel(filepath + "/AlexNet+CNN.xlsx")
    # data_RCNet = pd.read_excel(filepath + "/Vq-VAE-resnet18仅重构+分类器-71.xlsx")
    data_SRCNet = pd.read_excel(filepath + "/VQ-VAE-resnet18-29.xlsx")
    data_TSRCNet = pd.read_excel("../models2/筛查重构+分类_excels/筛查重构+分类-89.xlsx")

    # y_true_AlexNet = data_AlexNet.loc[:, "label"]
    # y_true_DenseNet = data_DenseNet.loc[:, "label"]
    # y_true_MobileNet = data_MobileNet.loc[:, "label"]
    # y_true_Resnet18 = data_Resnet18.loc[:, "label"]
    # y_true_Resnet50 = data_Resnet50.loc[:, "label"]
    # y_true_RCNet = data_RCNet.loc[:, "label"]
    y_true_SRCNet = data_SRCNet.loc[:, "label"]
    y_true_TSRCNet = data_TSRCNet.loc[:, "label"]

    # y_scores_AlexNet = data_AlexNet.loc[:, "pred"]
    # y_scores_DenseNet = data_DenseNet.loc[:, "pred"]
    # y_scores_MobileNet = data_MobileNet.loc[:, "pred"]
    # y_scores_Resnet18 = data_Resnet18.loc[:, "pred"]
    # y_scores_Resnet50 = data_Resnet50.loc[:, "pred"]
    # y_scores_RCNet = data_RCNet.loc[:, "pred"]
    y_scores_SRCNet = data_SRCNet.loc[:, "pred"]
    y_scores_TSRCNet = data_TSRCNet.loc[:, "pred"]

    # precision_AlexNet, recall_AlexNet, thresholds_AlexNet = precision_recall_curve(y_true_AlexNet, y_scores_AlexNet)
    # precision_DenseNet, recall_DenseNet, thresholds_DenseNet = precision_recall_curve(y_true_DenseNet, y_scores_DenseNet)
    # precision_MobileNet, recall_MobileNet, thresholds_MobileNet = precision_recall_curve(y_true_MobileNet, y_scores_MobileNet)
    # precision_Resnet18, recall_Resnet18, thresholds_Resnet18 = precision_recall_curve(y_true_Resnet18, y_scores_Resnet18)
    # precision_Resnet50, recall_Resnet50, thresholds_Resnet50 = precision_recall_curve(y_true_Resnet50, y_scores_Resnet50)
    # precision_RCNet, recall_RCNet, thresholds_RCNet = precision_recall_curve(y_true_RCNet, y_scores_RCNet)
    precision_SRCNet, recall_SRCNet, thresholds_SRCNet = precision_recall_curve(y_true_SRCNet, y_scores_SRCNet)
    precision_TSRCNet, recall_TSRCNet, thresholds_TSRCNet = precision_recall_curve(y_true_TSRCNet, y_scores_TSRCNet)

    # plt.plot(precision_Resnet18, recall_Resnet18, label="Resnet18")
    # plt.plot(precision_Resnet50, recall_Resnet50, label="Resnet50")
    # plt.plot(precision_AlexNet, recall_AlexNet, label="AlexNet")
    # plt.plot(precision_MobileNet, recall_MobileNet, label="MobileNet")
    # plt.plot(precision_DenseNet, recall_DenseNet, label="DenseNet")
    # plt.plot(precision_RCNet, recall_RCNet, label="RCNet")
    plt.plot(precision_SRCNet, recall_SRCNet, label="SRCNet")
    plt.plot(precision_TSRCNet, recall_TSRCNet, label="TSRCNet")

    plt.legend(loc="upper right")
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
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
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

    plt.ylabel('True label',fontsize=14)
    plt.xlabel('Predicted label',fontsize=14)
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
    data = pd.read_excel("../models3/excels2/Resnet50-73.xlsx")
    y_true = data.loc[:,"label"]
    y_prob = data.loc[:,"prob"]
    y_pred = data.loc[:,"pred"]
    # plot_confusion_matrix(y_true,y_prob)
    # PR_Curve(y_true,y_pred)
    # ROC_Curve(y_true,y_pred)

    compute(y_true,y_prob,y_pred)
    # Multi_PR_Curve()
    # Multi_ROC_Curve()