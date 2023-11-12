import os
import torch.utils.data as data
from PIL import Image


class mydataset(data.Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(image_dir)
        self.label_path = os.listdir(label_dir)

    def __getitem__(self, index):
        image_name = self.image_path[index]
        label_name = self.label_path[index]
        image_item_path = os.path.join(self.image_dir, image_name)
        label_item_path = os.path.join(self.label_dir, label_name)
        image = Image.open(image_item_path)
        label = label_item_path[index]
        return image, label

    def __len__(self):
        return len(self.label_path)


image_dir = 'D:\\File\\YOLO\\DL\\dataset\\coco128\\images\\train2017'
label_dir = 'D:\\File\\YOLO\\DL\\dataset\\coco128\\labels\\train2017'
dataset = mydataset(image_dir, label_dir)
