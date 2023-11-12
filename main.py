# import torch.cuda
import argparse
from utils.datasets import *
from torch.utils.data import DataLoader


def main():
    # 选择device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')

    # args = parser.parse_args()
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    data_folder = 'D:\\File\\YOLO\\datasets\\mnist\\train'
    datasets = LoadImagesAndLabels(data_folder)
    test_loader = DataLoader(dataset=datasets,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True)
    print("there are total %s batches for test" % (len(test_loader)))
    opt = parse_opt()
    main()
    # # 测试数据集中第一张图片
    # img, label = datasets[6000*5]
    # # img.show()    # 转为Tensor就不能show
    # print(img.shape)
    # print(label)
    #
    for data in test_loader:
        imgs, labels = data
        print(imgs.shape)
        print(labels)