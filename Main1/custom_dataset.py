"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

import pandas as pd
import numpy as np
import torch
from PIL import Image

import moco.loader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        normalize,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        # mean = [0.485, 0.456, 0.406]
        # std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([moco.loader.TwoCropsTransform(transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Compose(color_transform),
                transforms.ToTensor(),
                normalize]))
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops, target


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort



""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        target = self.dataset.targets[index]

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        # anchor['image'] = self.anchor_transform(anchor['image'])
        # neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = self.anchor_transform(anchor[0])
        output['neighbor'] = self.neighbor_transform(neighbor[0])
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = target
        
        return output

class CustomTwoViewDataset(Dataset):
    '''
    @param branch_pic:决定输入的双分支图像第二分支的类型。original 表示第二分支为第二波段原图
                                                    minus 表示使用相减后的图
    '''
    def __init__(self, data_dir1, data_dir2, transform=None, branch_pic='original'):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.transform = transform
        self.branch_pic = branch_pic
        self.data1 = ImageFolder(self.data_dir1, transform=transform)
        self.data2 = ImageFolder(self.data_dir2, transform=transform)

    def __len__(self):
        # 两个数据集的长度应该一样
        assert len(self.data1) == len(self.data2)
        return len(self.data1)

    def __getitem__(self, idx):
        # 获取相同索引的两组数据
        img1, label1 = self.data1[idx]
        img2, _ = self.data2[idx]
        if self.branch_pic == 'original':
            return [img1, img2], label1
        elif self.branch_pic == 'minus':
            return [img1, img1 - img2], label1
        elif self.branch_pic == 'concat':
            pass


class CustomThreeChannelDataset(Dataset):
    '''
    @param branch_pic:决定输入的双分支图像第二分支的类型。original 表示第二分支为第二波段原图
                                                    minus 表示使用相减后的图
    '''
    def __init__(self, data_dir1, data_dir2, data_dir3, transform=None):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.transform = transform
        self.data1 = ImageFolder(self.data_dir1, transform=transform)
        self.data2 = ImageFolder(self.data_dir2, transform=transform)
        self.data3 = ImageFolder(self.data_dir3, transform=transform)

    def __len__(self):
        # 两个数据集的长度应该一样
        assert len(self.data1) == len(self.data2)
        assert len(self.data1) == len(self.data3)
        return len(self.data1)

    def __getitem__(self, idx):
        # 获取相同索引的两组数据
        img1, label1 = self.data1[idx]
        img2, _ = self.data2[idx]
        img3, _ = self.data3[idx]
        return [img1, img2, img3, img1 - img2, img2 - img1], label1

class CustomDataset(Dataset):
    def __init__(self, data_dir, csv_dir, transform=None):
        self.data_dir = data_dir
        self.csv = pd.read_csv(csv_dir, index_col=0)
        self.transform = transform
        self.data = ImageFolder(self.data_dir, transform=transform)

    def __len__(self):
        # 两个数据集的长度应该一样
        return len(self.data)

    def __getitem__(self, idx):
        # 获取相同索引的两组数据
        img, label = self.data[idx]
        project_name = self.csv.loc[self.data.samples[idx][0].split('/')[-1]].project_name
        return img, label, project_name


def get_neighbor_dataset(traindir, transform, cluster_path, num_neighbors):
    dataset = datasets.ImageFolder(
        traindir, transform
    )
    indices = np.load(cluster_path)
    dataset = NeighborsDataset(dataset, indices, num_neighbors)

    return dataset

class PairedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.enum_dict = {
            'benign': 0 , 'benign': 1
        }
        self.samples = self._load_samples()
    def _load_samples(self):
        samples = {}
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name + '/images')
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    key = file_name[:-6]
                    if key not in samples:
                        samples[key] = [[], self.enum_dict[class_name]]
                    samples[key][0].append(os.path.join(class_dir, file_name))
        self.keys = list(samples.keys())
        return samples

    def __len__(self):
        return len(self.keys)  # 假设每个样本有两个视角

    def __getitem__(self, idx):
        # 找到对应的样本
        key = self.keys[idx]
        file1 = self.samples[key][0][0]
        file2 = self.samples[key][0][1]
        file3 = self.samples[key][0][2]
        class1 = self.samples[key][1]

        image1 = Image.open(file1).convert('RGB')
        image2 = Image.open(file2).convert('RGB')
        image3 = Image.open(file3).convert('RGB')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

        return [image1, image2, image3], class1