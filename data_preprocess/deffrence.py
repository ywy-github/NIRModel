import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def get_data(input_path, label):
    data = []
    labels = []

    for folder in ['benign', 'benign']:
        folder_path = os.path.join(input_path, folder)
        image_names = os.listdir(folder_path)

        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            img = Image.open(image_path)
            img = np.array(img)
            img_flattened = img.flatten()

            data.append(img_flattened)
            labels.append(label)

    return np.array(data), np.array(labels)

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    return fig
def plot_embedding_3D(data,label,title):
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    ax = plt.figure().add_subplot(111,projection='3d')
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9})
    return ax

def main():
    data_test, labels_test = get_data('../data/一期数据/test', label=0)
    data_val, labels_val = get_data('../data/筛查有病理数据_一期_52例/test', label=1)

    data_combined = np.concatenate((data_test, data_val), axis=0)
    labels_combined = np.concatenate((labels_test, labels_val), axis=0)

    print('Begining......')  # 时间会较长，所有处理完毕后给出finished提示
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    result_2D = tsne_2D.fit_transform(data_combined)
    tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    result_3D = tsne_3D.fit_transform(data_combined)
    print('Finished......')
    # 调用上面的两个函数进行可视化
    # fig1 = plot_embedding_2D(result_2D, labels_combined, 't-SNE')
    # fig1.show()
    fig2 = plot_embedding_3D(result_3D, labels_combined, 't-SNE')
    plt.show()
if __name__ == '__main__':
    main()
