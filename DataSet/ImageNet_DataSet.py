# coding: utf-8
from torchvision.datasets import ImageFolder
from utils.image import resize_image_without_annotation
import cv2
class ImageNet(ImageFolder):

    def __init__(self, root, transform=None, target_size=448):
        super(ImageNet, self).__init__(root)
        self.transform = transform
        self.idx_len = range(len(self))
        self.target_size = target_size

    def __getitem__(self, index):
        img_pth = self.imgs[index][0]
        label = self.imgs[index][1]
        img_data = cv2.imread(img_pth)
        img_data = resize_image_without_annotation(img_data, self.target_size, self.target_size)
        img_data = self.transform(img_data)
        return img_data, label
