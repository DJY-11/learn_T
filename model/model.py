import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

# '''下载并加载预训练的ResNet模型'''
# model = models.resnet18(pretrained=True)
#
# # 将模型保存到本地文件
# save_path = './model.pth'
# torch.save(model.state_dict(), save_path)

# mobilenet_v3_small1 = models.mobilenet_v3_small(weights=False)
# print('ok*******************')
# print(mobilenet_v3_small1)
# Model = mobilenet_v3_small1.classifier.add_module('Linear1', nn.Linear(1000, 10))
# print(mobilenet_v3_small2)
# torch.save(Model, "Model.pth")

# resnet = models.resnet18(weights=False)
# print(resnet)
# alexnet = models.alexnet()
# print(alexnet)
# vgg16 = models.vgg16()
# print(vgg16)
# squeezenet = models.squeezenet1_0()
# print(squeezenet)
from PIL import Image
from torchvision.models.resnet import conv3x3


class BasicBlock(nn.Module):

    expansion = 1   # 扩展因子，用于调整输出通道数，默认为1。

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        # norm_layer：归一化层的类型，默认为None。在初始化函数中，会创建卷积层、归一化层和激活函数等子模块，并将它们作为类的属性。
        # 批归一化（Batch Normalization）和组归一化（Group Normalization）等。这些归一化层可以在模型训练过程中对输入数据进行标准化处理，有助于提高模型的收敛速度和稳定性。
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x    # 保存输入的引用，用于与输出进行残差连接。
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

            out += identity     # 将 out 和 identity 相加，实现残差连接。
            out = self.relu(out)
            return out
#     def forward(self, x):
#         conv_layer1 = self.conv_layer1(x)
#         conv_layer2 = self.conv_layer2(conv_layer1)
#         conv_layer3 = self.conv_layer2(conv_layer2)
#         conv_layer4 = self.conv_layer2(conv_layer3)
#         FP = self.YourModule(conv_layer1,conv_layer2,conv_layer3,conv_layer4)
#         flatten = self.flatten(FN)
#         conn_layer1 = self.conn_layer1(flatten)
#         output = self.conn_layer2(conn_layer1)
#         return output


if __name__ == '__main__':
    basicblock = BasicBlock(inplanes=3, planes=64)
    print(basicblock)













