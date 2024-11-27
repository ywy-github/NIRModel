#通道注意力模块
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        act = self.sigmoid(avg_out + max_out)
        out = x * act
        return out

    #空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x1)
        x1 = self.sigmoid(x1)
        out = x * x1
        return out

    #先接通道注意力模块、再接空间注意力模块
class CBAM(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x*self.ca(x)
        result = out*self.sa(out)
        return x+result


class SKConv(nn.Module):
    def __init__(self, features, M: int = 2, G: int = 32, r: int = 16, stride: int = 1, L: int = 32):
        super().__init__()
        d = max(features / r, L)
        self.M = M
        self.features = features
        # 1.split
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
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
        batch_size = x.shape[0]
        # 1.split
        feats = torch.cat((x, y), dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        # 2.fuse
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        # 3.select
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)

        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V

class DS(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.point_conv = nn.Conv2d(input_channels,2*input_channels,1,1)
        self.Bn = nn.BatchNorm2d(2*input_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.point_conv(x)
        x = self.Bn(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    X = torch.randn(4, 64, 512, 512)
    model = DS(input_channels=64)  # 创建DS模块的实例，需要指定输入通道数
    outputs = model(X)  # 使用模块进行前向传播
