import os
from PIL import Image
import torch.utils.data as data


class my_dataset(data.Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.image_path[index]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir
        return image, label

    def __len__(self):
        return len(self.image_path)


root_dir = 'dataset/mnist/train'
label_dir = '1'
minist_dataset=my_dataset(root_dir, label_dir)

