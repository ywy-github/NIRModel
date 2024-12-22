import numpy as np
import pandas as pd
from imblearn.metrics import sensitivity_score, specificity_score
from matplotlib import pyplot as plt, rcParams
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
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    # plt.title('Receiver Operating Characteristic Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=18)
    plt.show()

def Multi_ROC_Curve_对比():
    filepath = "../models3/excels2"
    data_TransPath = pd.read_excel(filepath + "/TransPath.xlsx")
    data_TSBN = pd.read_excel(filepath + "/TSBN.xlsx")
    # data_MobileNet = pd.read_excel(filepath + "/MobileNet.xlsx")
    # data_Resnet18 = pd.read_excel(filepath + "/二期数据.xlsx")
    # data_Resnet50 = pd.read_excel(filepath + "/Resnet50.xlsx")
    data_SelfPath = pd.read_excel(filepath + "/SelfPath.xlsx")
    data_SSL = pd.read_excel(filepath + "/SSL.xlsx")
    data_our = pd.read_excel(filepath + "/our.xlsx")

    y_true_TransPath = data_TransPath.loc[:, "label"]
    y_true_TSBN = data_TSBN.loc[:, "label"]
    # y_true_MobileNet = data_MobileNet.loc[:, "label"]
    # y_true_Resnet18 = data_Resnet18.loc[:, "label"]
    # y_true_Resnet50 = data_Resnet50.loc[:, "label"]
    y_true_SelfPath = data_SelfPath.loc[:, "label"]
    y_true_SSL = data_SSL.loc[:, "label"]
    y_true_our = data_our.loc[:, "label"]

    y_scores_TransPath = data_TransPath.loc[:, "pred"]
    y_scores_TSBN = data_TSBN.loc[:, "pred"]
    # y_scores_MobileNet = data_MobileNet.loc[:, "pred"]
    # y_scores_Resnet18 = data_Resnet18.loc[:, "pred"]
    # y_scores_Resnet50 = data_Resnet50.loc[:, "pred"]
    y_scores_SelfPath = data_SelfPath.loc[:, "pred"]
    y_scores_SSL = data_SSL.loc[:, "pred"]
    y_scores_our = data_our.loc[:, "pred"]

    fpr_TransPath, tpr_TransPath, thresholds_TransPath = roc_curve(y_true_TransPath, y_scores_TransPath)
    fpr_TSBN, tpr_TSBN, thresholds_TSBN = roc_curve(y_true_TSBN,y_scores_TSBN)
    # fpr_MobileNet, tpr_MobileNet, thresholds_MobileNet = roc_curve(y_true_MobileNet,y_scores_MobileNet)
    # fpr_Resnet18, tpr_Resnet18, thresholds_Resnet18 = roc_curve(y_true_Resnet18,y_scores_Resnet18)
    # fpr_Resnet50, tpr_Resnet50, thresholds_Resnet50 = roc_curve(y_true_Resnet50,y_scores_Resnet50)
    fpr_SelfPath, tpr_SelfPath, thresholds_SelfPath = roc_curve(y_true_SelfPath, y_scores_SelfPath)
    fpr_SSL, tpr_SSL, thresholds_SSL = roc_curve(y_true_SSL, y_scores_SSL)
    fpr_our, tpr_our, thresholds_our = roc_curve(y_true_our, y_scores_our)

    roc_auc_TransPath = auc(fpr_TransPath, tpr_TransPath)
    roc_auc_TSBN = auc(fpr_TSBN, tpr_TSBN)
    # roc_auc_MobileNet = auc(fpr_MobileNet, tpr_MobileNet)
    # roc_auc_Resnet18 = auc(fpr_Resnet18, tpr_Resnet18)
    # roc_auc_Resnet50 = auc(fpr_Resnet50, tpr_Resnet50)
    roc_auc_SelfPath = auc(fpr_SelfPath, tpr_SelfPath)
    roc_auc_SSL = auc(fpr_SSL, tpr_SSL)
    roc_auc_our = auc(fpr_our, tpr_our)

    plt.figure()

    # plt.plot(fpr_Resnet18, tpr_Resnet18, linewidth=2,
    #          label='ResNet18 (area = %0.4f)' % roc_auc_Resnet18)
    # plt.plot(fpr_Resnet50, tpr_Resnet50, linewidth=2,
    #          label='ResNet50 (area = %0.4f)' % roc_auc_Resnet50)
    # plt.plot(fpr_MobileNet, tpr_MobileNet, linewidth=2,
    #          label='MobileNet (area = %0.4f)' % roc_auc_MobileNet)
    plt.plot(fpr_SelfPath, tpr_SelfPath, linewidth=2,
             label='Self-Path (area = %0.4f)' % roc_auc_SelfPath)
    plt.plot(fpr_TransPath, tpr_TransPath, linewidth=2,
             label='TransPath (area = %0.4f)' % roc_auc_TransPath)

    plt.plot(fpr_TSBN, tpr_TSBN, linewidth=2,
             label='TSBN (area = %0.4f)' % roc_auc_TSBN)

    plt.plot(fpr_SSL, tpr_SSL, linewidth=2,color='black',
             label='SSL (area = %0.4f)' % roc_auc_SSL)
    plt.plot(fpr_our, tpr_our, linewidth=2,color='red',
             label='the proposed model (area = %0.4f)' % roc_auc_our)

    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('Receiver Operating Characteristic Curve',fontsize=14)
    plt.legend(loc="lower right")
    plt.show()

def Multi_ROC_Curve_消融():
    filepath = "../models3/excels3"

    data_Resnet18 = pd.read_excel(filepath + "/Resnet18.xlsx")
    data_TResnet = pd.read_excel(filepath + "/TResnet.xlsx")
    data_SRCNet = pd.read_excel(filepath + "/SRCNet.xlsx")
    data_our = pd.read_excel(filepath + "/our.xlsx")

    y_true_Resnet18 = data_Resnet18.loc[:, "label"]
    y_true_TResnet = data_TResnet.loc[:, "label"]
    y_true_SRCNet = data_SRCNet.loc[:, "label"]
    y_true_our = data_our.loc[:, "label"]


    y_scores_Resnet18 = data_Resnet18.loc[:, "pred"]
    y_scores_TResnet = data_TResnet.loc[:, "pred"]
    y_scores_SRCNet = data_SRCNet.loc[:, "pred"]
    y_scores_our = data_our.loc[:, "pred"]


    fpr_Resnet18, tpr_Resnet18, thresholds_Resnet18 = roc_curve(y_true_Resnet18,y_scores_Resnet18)
    fpr_TResnet, tpr_TResnet, thresholds_TResnet = roc_curve(y_true_TResnet,y_scores_TResnet)
    fpr_SRCNet, tpr_SRCNet, thresholds_SRCNet = roc_curve(y_true_SRCNet, y_scores_SRCNet)
    fpr_our, tpr_our, thresholds_our = roc_curve(y_true_our, y_scores_our)

    roc_auc_Resnet18 = auc(fpr_Resnet18, tpr_Resnet18)
    roc_auc_TResnet = auc(fpr_TResnet, tpr_TResnet)
    roc_auc_SRCNet = auc(fpr_SRCNet, tpr_SRCNet)
    roc_auc_our = auc(fpr_our, tpr_our)

    plt.figure()

    rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.plot(fpr_Resnet18, tpr_Resnet18, linewidth=2,
             label='模型一 (AUC = %0.4f)' % roc_auc_Resnet18)

    plt.plot(fpr_TResnet, tpr_TResnet, linewidth=2,
             label='模型二 (AUC = %0.4f)' % roc_auc_TResnet)

    plt.plot(fpr_SRCNet, tpr_SRCNet, linewidth=2,
             label='模型三 (AUC = %0.4f)' % roc_auc_SRCNet)
    plt.plot(fpr_our, tpr_our, linewidth=2,color='red',
             label='模型四 (AUC = %0.4f)' % roc_auc_our)

    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('Receiver Operating Characteristic Curve',fontsize=14)
    plt.legend(loc="lower right")
    plt.show()


def PR_Curve(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')  # 显示P-R曲线
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')  # 将曲线下部分面积填充
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall', fontsize=18)
    plt.ylabel('precision', fontsize=18)
    # plt.title('precision-recall curve', fontsize=14)
    plt.show()

def Multi_PR_Curve():
    filepath = "../models2/excels2"
    data_SSL = pd.read_excel(filepath + "/SSL.xlsx")
    data_TSBN = pd.read_excel(filepath + "/TSBN.xlsx")
    data_MobileNet = pd.read_excel(filepath + "/MobileNet.xlsx")
    data_Resnet18 = pd.read_excel(filepath + "/二期数据.xlsx")
    data_Resnet50 = pd.read_excel(filepath + "/Resnet50.xlsx")
    data_SelfPath = pd.read_excel(filepath + "/SelfPath.xlsx.xlsx")
    data_TransPath = pd.read_excel(filepath + "/TransPath.xlsx")
    data_our = pd.read_excel(filepath + "/our.xlsx")

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
    plt.xticks(tick_marks, classes, rotation=45,fontsize=18)
    plt.yticks(tick_marks, classes,fontsize=18)

    # 在格子中显示准确率
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.4f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',fontsize=18)
    plt.xlabel('Predicted label',fontsize=18)
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

def all_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sen = np.round(sensitivity_score(y_true, y_pred), 4)
    spe = np.round(specificity_score(y_true, y_pred), 4)
    acc = np.round(accuracy_score(y_true, y_pred), 4)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]

    print("Acc: %.4f" % acc)
    print(("Sen: %.4f") % sen)
    print("Spec: %.4f" % spe)

if __name__ == '__main__':
    data = pd.read_excel("../models2/excels2/SSL.xlsx")
    y_true = data.loc[:,"label"]
    # y_prob = data.loc[:,"prob"]
    y_prob = data.loc[:,"prob"]
    # plot_confusion_matrix(y_true,y_prob)
    # PR_Curve(y_true,y_pred)
    # ROC_Curve(y_true,y_pred)

    # compute(y_true,y_prob,y_pred)

    Multi_ROC_Curve_消融()
    # Multi_ROC_Curve_对比()
    # all_metrics(y_true, y_prob)

