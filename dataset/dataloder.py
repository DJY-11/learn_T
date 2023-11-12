from utils.datasets import *
from torch.utils.data import DataLoader


data_folder = 'D:\\File\\YOLO\\datasets\\mnist\\train'

test_data = LoadImagesAndLabels(data_folder)
test_loader = DataLoader(dataset=test_data,
                         batch_size=64,
                         shuffle=True, 
                         num_workers=0, 
                         drop_last=True)

# 测试数据集中第一张图片
img, label = test_data[6000*5]
print(img.shape)
print(label)

for data in test_loader:
    imgs, labels = data
    print(imgs.shape)
    print(labels)