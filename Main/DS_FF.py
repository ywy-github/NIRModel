import torch
from torch import nn


class DS(nn.Module):
    def __init__(self, input_channels):
        super(DS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.point_conv = nn.Conv2d(input_channels,2*input_channels,1,1)
        self.Bn = nn.BatchNorm2d(2*input_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.point_conv(x)
        x = self.Bn(x)
        x = self.relu(x)
        return x

class FF(nn.Module):
    def __init__(self, features, M: int = 2, G: int = 32, r: int = 16, stride: int = 1, L: int = 32):
        super().__init__()
        d = max(features / r, L)
        self.M = M
        self.features = features
        # # 1.split
        # self.convs = nn.ModuleList([])
        # for i in range(M):
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
        #                   bias=False),
        #         nn.BatchNorm2d(features),
        #         nn.ReLU(inplace=True)
        #     ))
        # 2.fuse
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 3.select
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        feats_U = x+y
        # 2.fuse
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        # 3.select
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        print('attention_vectors.shape', attention_vectors.shape)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        print('attention_vectors.shape', attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        print('feats_V.shape', feats_V.shape)
        return feats_V


if __name__ == '__main__':
    inputs = torch.randn(4, 64, 512, 512)
    net = SKConv(64)
    outputs = net(inputs)

