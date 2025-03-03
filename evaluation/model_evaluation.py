import numpy as np
import pandas as pd
from imblearn.metrics import sensitivity_score, specificity_score
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score


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
    filepath = "../models2/excels2"
    data_TransPath = pd.read_excel(filepath + "/TransPath.xlsx")
    data_TSBN = pd.read_excel(filepath + "/TSBN.xlsx")
    data_SelfPath = pd.read_excel(filepath + "/SelfPath.xlsx")
    data_SSL = pd.read_excel(filepath + "/SSL.xlsx")
    data_our = pd.read_excel(filepath + "/our.xlsx")

    y_true_TransPath = data_TransPath.loc[:, "label"]
    y_true_TSBN = data_TSBN.loc[:, "label"]
    y_true_SelfPath = data_SelfPath.loc[:, "label"]
    y_true_SSL = data_SSL.loc[:, "label"]
    y_true_our = data_our.loc[:, "label"]

    y_scores_TransPath = data_TransPath.loc[:, "pred"]
    y_scores_TSBN = data_TSBN.loc[:, "pred"]
    y_scores_SelfPath = data_SelfPath.loc[:, "pred"]
    y_scores_SSL = data_SSL.loc[:, "pred"]
    y_scores_our = data_our.loc[:, "pred"]

    fpr_TransPath, tpr_TransPath, thresholds_TransPath = roc_curve(y_true_TransPath, y_scores_TransPath)
    fpr_TSBN, tpr_TSBN, thresholds_TSBN = roc_curve(y_true_TSBN,y_scores_TSBN)
    fpr_SelfPath, tpr_SelfPath, thresholds_SelfPath = roc_curve(y_true_SelfPath, y_scores_SelfPath)
    fpr_SSL, tpr_SSL, thresholds_SSL = roc_curve(y_true_SSL, y_scores_SSL)
    fpr_our, tpr_our, thresholds_our = roc_curve(y_true_our, y_scores_our)

    roc_auc_TransPath = auc(fpr_TransPath, tpr_TransPath)
    roc_auc_TSBN = auc(fpr_TSBN, tpr_TSBN)
    roc_auc_SelfPath = auc(fpr_SelfPath, tpr_SelfPath)
    roc_auc_SSL = auc(fpr_SSL, tpr_SSL)
    roc_auc_our = auc(fpr_our, tpr_our)

    filepath2 = "../models3/excels2"
    data_TransPath2 = pd.read_excel(filepath2 + "/TransPath.xlsx")
    data_TSBN2 = pd.read_excel(filepath2 + "/TSBN.xlsx")
    data_SelfPath2 = pd.read_excel(filepath2 + "/SelfPath.xlsx")
    data_SSL2 = pd.read_excel(filepath2 + "/SSL.xlsx")
    data_our2 = pd.read_excel(filepath2 + "/our.xlsx")

    y_true_TransPath2 = data_TransPath2.loc[:, "label"]
    y_true_TSBN2 = data_TSBN2.loc[:, "label"]
    y_true_SelfPath2 = data_SelfPath2.loc[:, "label"]
    y_true_SSL2 = data_SSL2.loc[:, "label"]
    y_true_our2 = data_our2.loc[:, "label"]

    y_scores_TransPath2 = data_TransPath2.loc[:, "pred"]
    y_scores_TSBN2 = data_TSBN2.loc[:, "pred"]
    y_scores_SelfPath2 = data_SelfPath2.loc[:, "pred"]
    y_scores_SSL2 = data_SSL2.loc[:, "pred"]
    y_scores_our2 = data_our2.loc[:, "pred"]

    fpr_TransPath2, tpr_TransPath2, thresholds_TransPath2 = roc_curve(y_true_TransPath2, y_scores_TransPath2)
    fpr_TSBN2, tpr_TSBN2, thresholds_TSBN2 = roc_curve(y_true_TSBN2, y_scores_TSBN2)
    fpr_SelfPath2, tpr_SelfPath2, thresholds_SelfPath2 = roc_curve(y_true_SelfPath2, y_scores_SelfPath2)
    fpr_SSL2, tpr_SSL2, thresholds_SSL2 = roc_curve(y_true_SSL2, y_scores_SSL2)
    fpr_our2, tpr_our2, thresholds_our2 = roc_curve(y_true_our2, y_scores_our2)

    roc_auc_TransPath2 = auc(fpr_TransPath2, tpr_TransPath2)
    roc_auc_TSBN2 = auc(fpr_TSBN2, tpr_TSBN2)
    roc_auc_SelfPath2 = auc(fpr_SelfPath2, tpr_SelfPath2)
    roc_auc_SSL2 = auc(fpr_SSL2, tpr_SSL2)
    roc_auc_our2 = auc(fpr_our2, tpr_our2)


    rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列

    # 第一张图
    axes[0].plot(fpr_SelfPath, tpr_SelfPath, linewidth=2,
             label='Self-Path (AUC = %0.4f)' % roc_auc_SelfPath)

    axes[0].plot(fpr_TransPath, tpr_TransPath, linewidth=2,
             label='TransPath (AUC = %0.4f)' % roc_auc_TransPath)

    axes[0].plot(fpr_TSBN, tpr_TSBN, linewidth=2,
             label='TSBN (AUC = %0.4f)' % roc_auc_TSBN)

    axes[0].plot(fpr_SSL, tpr_SSL, linewidth=2,
                 label='SSL (AUC = %0.4f)' % roc_auc_SSL)

    axes[0].plot(fpr_our, tpr_our, linewidth=2, color='red',
             label='SRCNet (AUC = %0.4f)' % roc_auc_our)

    axes[0].set_xlabel('FPR', fontsize=16)
    axes[0].set_ylabel('TPR', fontsize=16)
    axes[0].set_title('ROC曲线(一期数据)', fontsize=16)
    axes[0].legend(loc="lower right", fontsize=12)


    # 第二张图
    axes[1].plot(fpr_SelfPath2, tpr_SelfPath2, linewidth=2,
                 label='Self-Path (AUC = %0.4f)' % roc_auc_SelfPath2)

    axes[1].plot(fpr_TransPath2, tpr_TransPath2, linewidth=2,
                 label='TransPath (AUC = %0.4f)' % roc_auc_TransPath2)

    axes[1].plot(fpr_TSBN2, tpr_TSBN2, linewidth=2,
                 label='TSBN (AUC = %0.4f)' % roc_auc_TSBN2)

    axes[1].plot(fpr_SSL2, tpr_SSL2, linewidth=2,
                 label='SSL (AUC = %0.4f)' % roc_auc_SSL2)

    axes[1].plot(fpr_our2, tpr_our2, linewidth=2, color='red',
                 label='SRCNet (AUC = %0.4f)' % roc_auc_our2)

    axes[1].set_xlabel('FPR', fontsize=16)
    axes[1].set_ylabel('TPR', fontsize=16)
    axes[1].set_title('ROC曲线(二期数据)', fontsize=16)
    axes[1].legend(loc="lower right", fontsize=12)

    plt.tight_layout()  # 调整布局
    plt.show()

    plt.savefig("../data/ROC/SRCNet对比.pdf")

def Multi_ROC_Curve_对比2():
    filepath = "../MultiScale/对比/data1"
    data_TransPath = pd.read_excel(filepath + "/FabNet.xlsx")
    data_TSBN = pd.read_excel(filepath + "/M2S2-FNet.xlsx")
    data_SelfPath = pd.read_excel(filepath + "/MDAA.xlsx")
    data_SSL = pd.read_excel(filepath + "/RI-ViT.xlsx")
    data_our = pd.read_excel(filepath + "/MSFEFNet.xlsx")

    y_true_TransPath = data_TransPath.loc[:, "label"]
    y_true_TSBN = data_TSBN.loc[:, "label"]
    y_true_SelfPath = data_SelfPath.loc[:, "label"]
    y_true_SSL = data_SSL.loc[:, "label"]
    y_true_our = data_our.loc[:, "label"]

    y_scores_TransPath = data_TransPath.loc[:, "pred"]
    y_scores_TSBN = data_TSBN.loc[:, "pred"]
    y_scores_SelfPath = data_SelfPath.loc[:, "pred"]
    y_scores_SSL = data_SSL.loc[:, "pred"]
    y_scores_our = data_our.loc[:, "pred"]

    fpr_TransPath, tpr_TransPath, thresholds_TransPath = roc_curve(y_true_TransPath, y_scores_TransPath)
    fpr_TSBN, tpr_TSBN, thresholds_TSBN = roc_curve(y_true_TSBN,y_scores_TSBN)
    fpr_SelfPath, tpr_SelfPath, thresholds_SelfPath = roc_curve(y_true_SelfPath, y_scores_SelfPath)
    fpr_SSL, tpr_SSL, thresholds_SSL = roc_curve(y_true_SSL, y_scores_SSL)
    fpr_our, tpr_our, thresholds_our = roc_curve(y_true_our, y_scores_our)

    roc_auc_TransPath = auc(fpr_TransPath, tpr_TransPath)
    roc_auc_TSBN = auc(fpr_TSBN, tpr_TSBN)
    roc_auc_SelfPath = auc(fpr_SelfPath, tpr_SelfPath)
    roc_auc_SSL = auc(fpr_SSL, tpr_SSL)
    roc_auc_our = auc(fpr_our, tpr_our)

    filepath2 = "../MultiScale/对比/data2"
    data_TransPath2 = pd.read_excel(filepath2 + "/FabNet.xlsx")
    data_TSBN2 = pd.read_excel(filepath2 + "/M2S2-FNet.xlsx")
    data_SelfPath2 = pd.read_excel(filepath2 + "/MDAA.xlsx")
    data_SSL2 = pd.read_excel(filepath2 + "/RI-ViT.xlsx")
    data_our2 = pd.read_excel(filepath2 + "/MSFEFNet.xlsx")

    y_true_TransPath2 = data_TransPath2.loc[:, "label"]
    y_true_TSBN2 = data_TSBN2.loc[:, "label"]
    y_true_SelfPath2 = data_SelfPath2.loc[:, "label"]
    y_true_SSL2 = data_SSL2.loc[:, "label"]
    y_true_our2 = data_our2.loc[:, "label"]

    y_scores_TransPath2 = data_TransPath2.loc[:, "pred"]
    y_scores_TSBN2 = data_TSBN2.loc[:, "pred"]
    y_scores_SelfPath2 = data_SelfPath2.loc[:, "pred"]
    y_scores_SSL2 = data_SSL2.loc[:, "pred"]
    y_scores_our2 = data_our2.loc[:, "pred"]

    fpr_TransPath2, tpr_TransPath2, thresholds_TransPath2 = roc_curve(y_true_TransPath2, y_scores_TransPath2)
    fpr_TSBN2, tpr_TSBN2, thresholds_TSBN2 = roc_curve(y_true_TSBN2, y_scores_TSBN2)
    fpr_SelfPath2, tpr_SelfPath2, thresholds_SelfPath2 = roc_curve(y_true_SelfPath2, y_scores_SelfPath2)
    fpr_SSL2, tpr_SSL2, thresholds_SSL2 = roc_curve(y_true_SSL2, y_scores_SSL2)
    fpr_our2, tpr_our2, thresholds_our2 = roc_curve(y_true_our2, y_scores_our2)

    roc_auc_TransPath2 = auc(fpr_TransPath2, tpr_TransPath2)
    roc_auc_TSBN2 = auc(fpr_TSBN2, tpr_TSBN2)
    roc_auc_SelfPath2 = auc(fpr_SelfPath2, tpr_SelfPath2)
    roc_auc_SSL2 = auc(fpr_SSL2, tpr_SSL2)
    roc_auc_our2 = auc(fpr_our2, tpr_our2)


    rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列

    # 第一张图
    axes[0].plot(fpr_SelfPath, tpr_SelfPath, linewidth=2,
             label='MDAA (AUC = %0.4f)' % roc_auc_SelfPath)

    axes[0].plot(fpr_TransPath, tpr_TransPath, linewidth=2,
             label='FabNet (AUC = %0.4f)' % roc_auc_TransPath)

    axes[0].plot(fpr_TSBN, tpr_TSBN, linewidth=2,
             label='M2S2-FNet (AUC = %0.4f)' % roc_auc_TSBN)

    axes[0].plot(fpr_SSL, tpr_SSL, linewidth=2,
                 label='RI-ViT (AUC = %0.4f)' % roc_auc_SSL)

    axes[0].plot(fpr_our, tpr_our, linewidth=2, color='red',
             label='MSFEFNet (AUC = %0.4f)' % roc_auc_our)

    axes[0].set_xlabel('FPR', fontsize=16)
    axes[0].set_ylabel('TPR', fontsize=16)
    axes[0].set_title('ROC曲线(一期数据)', fontsize=16)
    axes[0].legend(loc="lower right", fontsize=12)


    # 第二张图
    axes[1].plot(fpr_SelfPath2, tpr_SelfPath2, linewidth=2,
                 label='MDAA (AUC = %0.4f)' % roc_auc_SelfPath2)

    axes[1].plot(fpr_TransPath2, tpr_TransPath2, linewidth=2,
                 label='FabNet (AUC = %0.4f)' % roc_auc_TransPath2)

    axes[1].plot(fpr_TSBN2, tpr_TSBN2, linewidth=2,
                 label='M2S2-FNet (AUC = %0.4f)' % roc_auc_TSBN2)

    axes[1].plot(fpr_SSL2, tpr_SSL2, linewidth=2,
                 label='RI-ViT (AUC = %0.4f)' % roc_auc_SSL2)

    axes[1].plot(fpr_our2, tpr_our2, linewidth=2, color='red',
                 label='MSFEFNet (AUC = %0.4f)' % roc_auc_our2)

    axes[1].set_xlabel('FPR', fontsize=16)
    axes[1].set_ylabel('TPR', fontsize=16)
    axes[1].set_title('ROC曲线(二期数据)', fontsize=16)
    axes[1].legend(loc="lower right", fontsize=12)

    plt.tight_layout()  # 调整布局
    plt.show()

    plt.savefig("../data/ROC/MSFEFNet.pdf")


def Multi_ROC_Curve_消融():
    filepath = "../models2/excels3"

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

    filepath2 = "../models3/excels3"

    data_Resnet182 = pd.read_excel(filepath2 + "/Resnet18.xlsx")
    data_TResnet2 = pd.read_excel(filepath2 + "/TResnet.xlsx")
    data_SRCNet2 = pd.read_excel(filepath2 + "/SRCNet.xlsx")
    data_our2 = pd.read_excel(filepath2 + "/our.xlsx")

    y_true_Resnet182 = data_Resnet182.loc[:, "label"]
    y_true_TResnet2 = data_TResnet2.loc[:, "label"]
    y_true_SRCNet2 = data_SRCNet2.loc[:, "label"]
    y_true_our2 = data_our2.loc[:, "label"]

    y_scores_Resnet182 = data_Resnet182.loc[:, "pred"]
    y_scores_TResnet2 = data_TResnet2.loc[:, "pred"]
    y_scores_SRCNet2 = data_SRCNet2.loc[:, "pred"]
    y_scores_our2 = data_our2.loc[:, "pred"]

    fpr_Resnet182, tpr_Resnet182, thresholds_Resnet182 = roc_curve(y_true_Resnet182, y_scores_Resnet182)
    fpr_TResnet2, tpr_TResnet2, thresholds_TResnet2 = roc_curve(y_true_TResnet2, y_scores_TResnet2)
    fpr_SRCNet2, tpr_SRCNet2, thresholds_SRCNet2 = roc_curve(y_true_SRCNet2, y_scores_SRCNet2)
    fpr_our2, tpr_our2, thresholds_our2 = roc_curve(y_true_our2, y_scores_our2)

    roc_auc_Resnet182 = auc(fpr_Resnet182, tpr_Resnet182)
    roc_auc_TResnet2 = auc(fpr_TResnet2, tpr_TResnet2)
    roc_auc_SRCNet2 = auc(fpr_SRCNet2, tpr_SRCNet2)
    roc_auc_our2 = auc(fpr_our2, tpr_our2)



    rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列

    # 第一张图
    axes[0].plot(fpr_Resnet18, tpr_Resnet18, linewidth=2,
             label='w/o Both (AUC = %0.4f)' % roc_auc_Resnet18)

    axes[0].plot(fpr_TResnet, tpr_TResnet, linewidth=2,
             label='w/o 1 (AUC = %0.4f)' % roc_auc_TResnet)

    axes[0].plot(fpr_SRCNet, tpr_SRCNet, linewidth=2,
             label='w/o 2 (AUC = %0.4f)' % roc_auc_SRCNet)

    axes[0].plot(fpr_our, tpr_our, linewidth=2,color='red',
             label='SRCNet (AUC = %0.4f)' % roc_auc_our)

    axes[0].set_xlabel('FPR', fontsize=16)
    axes[0].set_ylabel('TPR', fontsize=16)
    axes[0].set_title('ROC曲线(一期数据)', fontsize=16)
    axes[0].legend(loc="lower right", fontsize=12)


    # 第二张图
    axes[1].plot(fpr_Resnet182, tpr_Resnet182, linewidth=2,
                 label='w/o Both (AUC = %0.4f)' % roc_auc_Resnet182)

    axes[1].plot(fpr_TResnet2, tpr_TResnet2, linewidth=2,
                 label='w/o 1 (AUC = %0.4f)' % roc_auc_TResnet2)

    axes[1].plot(fpr_SRCNet2, tpr_SRCNet2, linewidth=2,
                 label='w/o 2 (AUC = %0.4f)' % roc_auc_SRCNet2)

    axes[1].plot(fpr_our2, tpr_our2, linewidth=2, color='red',
                 label='SRCNet (AUC = %0.4f)' % roc_auc_our2)

    axes[1].set_xlabel('FPR', fontsize=16)
    axes[1].set_ylabel('TPR', fontsize=16)
    axes[1].set_title('ROC曲线(二期数据)', fontsize=16)
    axes[1].legend(loc="lower right", fontsize=12)

    plt.tight_layout()  # 调整布局
    plt.show()

    plt.savefig("../data/ROC/SRCNet消融.pdf")


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

def Multi_ROC_Curve_消融2():
    filepath = "../MultiScale/消融/data1"

    data_Resnet18 = pd.read_excel(filepath + "/1.xlsx")
    data_TResnet = pd.read_excel(filepath + "/2.xlsx")
    data_SRCNet = pd.read_excel(filepath + "/Both.xlsx")
    data_our = pd.read_excel(filepath + "/MSFEFNet.xlsx")

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

    filepath2 = "../MultiScale/消融/data2"
    data_Resnet182 = pd.read_excel(filepath2 + "/1.xlsx")
    data_TResnet2 = pd.read_excel(filepath2 + "/2.xlsx")
    data_SRCNet2 = pd.read_excel(filepath2 + "/Both.xlsx")
    data_our2 = pd.read_excel(filepath2 + "/MSFEFNet.xlsx")

    y_true_Resnet182 = data_Resnet182.loc[:, "label"]
    y_true_TResnet2 = data_TResnet2.loc[:, "label"]
    y_true_SRCNet2 = data_SRCNet2.loc[:, "label"]
    y_true_our2 = data_our2.loc[:, "label"]

    y_scores_Resnet182 = data_Resnet182.loc[:, "pred"]
    y_scores_TResnet2 = data_TResnet2.loc[:, "pred"]
    y_scores_SRCNet2 = data_SRCNet2.loc[:, "pred"]
    y_scores_our2 = data_our2.loc[:, "pred"]

    fpr_Resnet182, tpr_Resnet182, thresholds_Resnet182 = roc_curve(y_true_Resnet182, y_scores_Resnet182)
    fpr_TResnet2, tpr_TResnet2, thresholds_TResnet2 = roc_curve(y_true_TResnet2, y_scores_TResnet2)
    fpr_SRCNet2, tpr_SRCNet2, thresholds_SRCNet2 = roc_curve(y_true_SRCNet2, y_scores_SRCNet2)
    fpr_our2, tpr_our2, thresholds_our2 = roc_curve(y_true_our2, y_scores_our2)

    roc_auc_Resnet182 = auc(fpr_Resnet182, tpr_Resnet182)
    roc_auc_TResnet2 = auc(fpr_TResnet2, tpr_TResnet2)
    roc_auc_SRCNet2 = auc(fpr_SRCNet2, tpr_SRCNet2)
    roc_auc_our2 = auc(fpr_our2, tpr_our2)

    rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列

    # 第一张图
    axes[0].plot(fpr_Resnet18, tpr_Resnet18, linewidth=2,
                 label='w/o Both (AUC = %0.4f)' % roc_auc_Resnet18)

    axes[0].plot(fpr_TResnet, tpr_TResnet, linewidth=2,
                 label='w/o 1 (AUC = %0.4f)' % roc_auc_TResnet)

    axes[0].plot(fpr_SRCNet, tpr_SRCNet, linewidth=2,
                 label='w/o 2 (AUC = %0.4f)' % roc_auc_SRCNet)

    axes[0].plot(fpr_our, tpr_our, linewidth=2, color='red',
                 label='TSRCNet (AUC = %0.4f)' % roc_auc_our)

    axes[0].set_xlabel('FPR', fontsize=16)
    axes[0].set_ylabel('TPR', fontsize=16)
    axes[0].set_title('ROC曲线(一期数据)', fontsize=16)
    axes[0].legend(loc="lower right", fontsize=12)

    # 第二张图
    axes[1].plot(fpr_Resnet182, tpr_Resnet182, linewidth=2,
                 label='w/o Both (AUC = %0.4f)' % roc_auc_Resnet182)

    axes[1].plot(fpr_TResnet2, tpr_TResnet2, linewidth=2,
                 label='w/o 1 (AUC = %0.4f)' % roc_auc_TResnet2)

    axes[1].plot(fpr_SRCNet2, tpr_SRCNet2, linewidth=2,
                 label='w/o 2 (AUC = %0.4f)' % roc_auc_SRCNet2)

    axes[1].plot(fpr_our2, tpr_our2, linewidth=2, color='red',
                 label='TSRCNet (AUC = %0.4f)' % roc_auc_our2)

    axes[1].set_xlabel('FPR', fontsize=16)
    axes[1].set_ylabel('TPR', fontsize=16)
    axes[1].set_title('ROC曲线(二期数据)', fontsize=16)
    axes[1].legend(loc="lower right", fontsize=12)

    plt.tight_layout()  # 调整布局
    plt.show()

    plt.savefig("../data/ROC/MSFEFNet消融.pdf")

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

def all_metrics(y_true, y_prob, y_pred):
    cm = confusion_matrix(y_true, y_prob)
    sen = np.round(sensitivity_score(y_true, y_prob), 4)
    spe = np.round(specificity_score(y_true, y_prob), 4)
    acc = np.round(accuracy_score(y_true, y_prob), 4)

    train_auc = roc_auc_score(y_true, y_pred)

    print("Acc: %.4f" % acc)
    print(("Sen: %.4f") % sen)
    print("Spec: %.4f" % spe)
    print("AUC: %.4f" % train_auc)

if __name__ == '__main__':
    data = pd.read_excel("../MultiScale/消融/data2/MSFEFNet.xlsx")
    y_true = data.loc[:,"label"]
    y_pred = data.loc[:,"pred"]
    y_prob = data.loc[:,"prob"]
    # plot_confusion_matrix(y_true,y_prob)
    # PR_Curve(y_true,y_pred)
    # ROC_Curve(y_true,y_pred)

    # compute(y_true,y_prob,y_pred)
    # Multi_ROC_Curve_消融2()
    # Multi_ROC_Curve_消融()
    # Multi_ROC_Curve_对比()
    Multi_ROC_Curve_对比2()
    # all_metrics(y_true, y_prob, y_pred)

