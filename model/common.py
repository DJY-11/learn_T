import math
import torch
import torch.nn as nn
from torch.nn import Flatten


# class YOLO_v1(nn.Module):
#     def __init__(self, num_class):
#         super(YOLO_v1, self).__init__()
#         c = num_class
#         self.conv_layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=7//2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1),  # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         """
#         BatchNorm2d(64):
#         num_features：输入图像的通道数量-C;
#         eps：稳定系数，防止分母出现0;
#         momentum：BatchNorm2d里面存储均值（running_mean）和方差（running_var）更新时的参数
#         affine：代表gamma，beta是否可学。如果设为True，代表两个参数是通过学习得到的；如果设为False，代表两个参数是固定值，默认情况下，gamma是1，beta是0
#         track_running_stats：BatchNorm2d中存储的的均值和方差是否需要更新，若为True，表示需要更新；反之不需要更新
#         """
#         self.conv_layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3//2),
#             nn.BatchNorm2d(192),
#             nn.LeakyReLU(0.1),  # negative_slope:x为负数时的需要的一个系数，控制负斜率的角度。默认值：1e-2; inplace：可以选择就地执行操作。默认值：False
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.flatten = Flatten()
#
#         self.conn_layer1 = nn.Sequential(
#             nn.Linear(in_features=7*7*1024, out_features=4096),
#             nn.Dropout(0.5),    # 3.nn.Dropout(p = 0.5) # 表示每个神经元有0.5的可能性不被激活
#             nn.LeakyReLU(0.1)
#         )
#         self.conn_layer2 = nn.Sequential(
#             nn.Linear(in_features=4096, out_features=7*7*(2*5+c))
#         )
#         self._initialize_weights()
#
#     def forward(self, input):
#         conv_layer1 = self.conn_layer1(input)
#         conv_layer2 = self.conv_layer2(conv_layer1)
#         flatten = self.flatten(conv_layer2)
#         conn_layer1 = self.conn_layer1(flatten)
#         output = self.conn_layer2(conn_layer1)
#         return output
#
#     def _initialize_weights(self):  # 初始化神经网络层的权重
#         for m in self.modules():    # 遍历神经网络中的所有模块
#
#             "isinstance"    # 判断一个函数是否是一个已知的类型
#             if isinstance(m, nn.Conv2d):    # 检查当前模块是否为二维卷积层 (nn.Conv2d)
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels    # 计算卷积层的输入通道数。
#                 m.weight.data.normal_(0, math.sqrt(2. / n))     # 使用均值为 0、标准差为 math.sqrt(2. / n) 的正态分布初始化卷积层的权重。
#                 if m.bias is not None:  # 检查卷积层是否有偏置项。
#                     m.bias.data.zero_()     # 将卷积层的偏置项初始化为零。
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)  # 将批归一化层的缩放因子初始化为 1。
#                 m.bias.data.zero_()     # 将批归一化层的偏置项初始化为零。
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)  # 用均值为 0、标准差为 0.01 的正态分布初始化线性层的权重。
#                 m.bias.data.zero_()     # 将线性层的偏置项初始化为零。
#     """
#     总结起来，这个方法用于初始化神经网络中卷积层、批归一化层和线性层的权重和偏置项。
#     权重使用不同的策略进行初始化，例如使用特定参数的正态分布，而偏置项都被初始化为零。
#     这些初始化技术有助于神经网络的训练过程。
#     """
#
#
# if __name__ == '__main__':
#     model = YOLO_v1(num_class=10)
#     print(model)


"""给网络添加自定义层"""
from torchvision import models


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def __forward(self, x):
        return x.view(x.size(0), -1)


class yournet(nn.Module):
    def __init__(self, stride=2, pool="avg"):
        super(yournet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.load_state_dict(torch.load('D:\\File\\YOLO\\DL\\model\\resnet50.pth'))
        self.flatten = Flatten()
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1)
        )
        self.conn_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=7*7*(2*5+20))
        )

    def __forward(self, x):
        x = self.resnet50(x)
        x = self.flatten(x)
        x = self.conn_layer1(x)
        x = self.conn_layer2(x)
        return x


if __name__ == "__main__":
    model = yournet(stride=2, pool="avg")
    print(model)
#     pretrained_dict = torch.load('D:\\File\\YOLO\\DL\\model\\resnet50.pth')
#     NO_W = pretrained_dict.pop('fc.weight')
#     pretrained_dict.pop('fc.bias')
#     model_dict = model.state_dict()
#     print(model_dict)
#     print(NO_W)
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}