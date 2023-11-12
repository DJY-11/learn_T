import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data


class LoadImagesAndLabels(data.Dataset):
    # def __init__(self, data_folder, opt):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.filenames = []
        self.labels = []
        # self.opt = opt
        
        per_classes = os.listdir(data_folder)   # 获取数据文件夹中的子目录（所有类别的文件夹）
        for per_class in per_classes:
            per_class_paths = os.path.join(data_folder, per_class)   # 构建当前类别目录的完整路径
            label = torch.tensor(int(per_class))

            per_datas = os.listdir(per_class_paths)     # 获取当前类别目录中的数据文件列表
            for per_data in per_datas:
                self.filenames.append(os.path.join(per_class_paths, per_data))   # 将数据文件的完整路径添加到文件名列表中
                self.labels.append(label)   # 将对应类别的标签添加到标签列表中
        
    def __getitem__(self, index):  # 通过index返回一个样本
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        data = self.proprecess(image)     # 进行proprecess后得到的数据
        return data, label
        # return image, label

    def __len__(self):
        return len(self.filenames)

    def proprecess(self, data):
        transform_train_list = [
            # transforms.Resize((self.opt.h, self.opt.w), interpolation=3),  # interpolation=3 指定在调整大小过程中使用的插值方法。在这种情况下，3 对应于 BILINEAR 插值方法。这意味着算法将使用双线性插值计算调整大小后图像的像素值。
            # transforms.Resize((self.opt.h, self.opt.w)),
            # transforms.Pad(self.opt.pad, padding_mode='edge'),
            # transforms.RandomCrop((self.opt.h, self.opt.w)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]

        data_transform = transforms.Compose(transform_train_list)
        return data_transform(data)
