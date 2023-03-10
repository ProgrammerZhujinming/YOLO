from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from utils import image

class VOC_Detection_Set(Dataset):
    def __init__(self, imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages",
                 annotations_path="../DataSet/VOC2007+2012/Train/Annotations",
                 classes_file="../DataSet/VOC2007+2012/class.data", is_train = True, class_num=20,
                 label_smooth_value=0.05, input_size=448, grid_size=64, loss_mode="mse"):  # input_size:输入图像的尺度
        self.label_smooth_value = label_smooth_value
        self.class_num = class_num
        self.imgs_name = os.listdir(imgs_path)
        self.input_size = input_size
        self.grid_size = grid_size
        self.is_train = is_train
        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        self.class_dict = {}
        self.loss_mode = loss_mode

        class_index = 0
        with open(classes_file, 'r') as file:
            for class_name in file:
                class_name = class_name.replace('\n', '')
                self.class_dict[class_name] = class_index  # 根据类别名制作索引
                class_index = class_index + 1

    def __getitem__(self, item):

        img_path = os.path.join(self.imgs_path, self.imgs_name[item])
        annotation_path = os.path.join(self.annotations_path, self.imgs_name[item].replace(".jpg", ".xml"))
        img = cv2.imread(img_path)
        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()

        objects_xml = annotation_xml.findall("object")
        coords = []

        for object_xml in objects_xml:
            bnd_xml = object_xml.find("bndbox")
            class_name = object_xml.find("name").text
            if class_name not in self.class_dict:  # 不属于我们规定的类
                continue
            xmin = round((float)(bnd_xml.find("xmin").text))
            ymin = round((float)(bnd_xml.find("ymin").text))
            xmax = round((float)(bnd_xml.find("xmax").text))
            ymax = round((float)(bnd_xml.find("ymax").text))
            class_id = self.class_dict[class_name]
            coords.append([xmin, ymin, xmax, ymax, class_id])

        coords.sort(key=lambda coord : (coord[2] - coord[0]) * (coord[3] - coord[1]) )

        if self.is_train:

            transform_seed = random.randint(0, 4)

            if transform_seed == 0:  # 原图
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 1:  # 缩放+中心裁剪
                img, coords = image.center_crop_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 2:  # 平移
                img, coords = image.transplant_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            else:  # 曝光度调整
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = image.exposure(img, gamma=0.5)
                img = self.transform_common(img)

        else:
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        ground_truth, ground_mask_positive, ground_mask_negative = self.getGroundTruth(coords)
        return img, [ground_truth, ground_mask_positive, ground_mask_negative, img_path]

        #ground_truth, ground_mask_positive, ground_mask_negative = self.getGroundTruth(coords)

        # 通道变化方法: img = img[:, :, ::-1]

        #return img, ground_truth, ground_mask_positive, ground_mask_negative

    def __len__(self):
        return len(self.imgs_name)

    def getGroundTruth(self, coords):

        feature_size = self.input_size // self.grid_size
        #ground_mask_positive = np.zeros([feature_size, feature_size, 1], dtype=bool)
        #ground_mask_negative = np.ones([feature_size, feature_size, 1], dtype=bool)
        ground_mask_positive = np.full(shape=(feature_size, feature_size, 1), fill_value=False, dtype=bool)
        ground_mask_negative = np.full(shape=(feature_size, feature_size, 1), fill_value=True, dtype=bool)

        if self.loss_mode == "mse":
            ground_truth = np.zeros([feature_size, feature_size, 10 + self.class_num + 2])
        else:
            ground_truth = np.zeros([feature_size, feature_size, 10 + 1])

        for coord in coords:

            xmin, ymin, xmax, ymax, class_id = coord

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            index_row = (int)(center_y * feature_size)
            index_col = (int)(center_x * feature_size)

            # 分类标签 label_smooth
            if self.loss_mode == "mse":
                # 转化为one_hot编码 对one_hot编码做平滑处理
                class_list = np.full(shape=self.class_num, fill_value=1.0, dtype=float)
                deta = 0.01
                class_list = class_list * deta / (self.class_num - 1)
                class_list[class_id] = 1.0 - deta
            elif self.loss_mode == "cross_entropy":
                class_list = [class_id]
            else:
                raise Exception("the loss mode can't be support now!")

            # 定位数据预设
            ground_box = [center_x * feature_size - index_col, center_y * feature_size - index_row,
                          ground_width, ground_height, 1,
                          round(xmin * self.input_size), round(ymin * self.input_size),
                          round(xmax * self.input_size), round(ymax * self.input_size),
                          round(ground_width * self.input_size * ground_height * self.input_size)
                          ]
            ground_box.extend(class_list)
            ground_box.extend([index_col, index_row])

            ground_truth[index_row][index_col] = np.array(ground_box)
            ground_mask_positive[index_row][index_col] = True
            ground_mask_negative[index_row][index_col] = False

        return ground_truth, torch.BoolTensor(ground_mask_positive), torch.BoolTensor(ground_mask_negative)

    #def test_read_data(self, img_path='./VOC2007+2012/Train/JPEGImages/2010_006984.jpg', annotation_path='./VOC2007+2012/Train/Annotations/2010_006984.xml'):
    #def test_read_data(self, img_path='./VOC2007+2012/Train/JPEGImages/2012_000084.jpg', annotation_path='./VOC2007+2012/Train/Annotations/2012_000084.xml'):
    #def test_read_data(self, img_path='./VOC2007+2012/Train/JPEGImages/2011_005723.jpg', annotation_path='./VOC2007+2012/Train/Annotations/2011_005723.xml'):
    #def test_read_data(self, img_path='./VOC2007+2012/Train/JPEGImages/2010_003491.jpg', annotation_path='./VOC2007+2012/Train/Annotations/2010_003491.xml'):
    def test_read_data(self, img_path='./VOC2007+2012/Train/JPEGImages/2010_006222.jpg', annotation_path='./VOC2007+2012/Train/Annotations/2010_006222.xml'):

        img = cv2.imread(img_path)
        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()

        objects_xml = annotation_xml.findall("object")
        coords = []

        for object_xml in objects_xml:
            bnd_xml = object_xml.find("bndbox")
            class_name = object_xml.find("name").text
            if class_name not in self.class_dict:  # 不属于我们规定的类
                continue
            xmin = round((float)(bnd_xml.find("xmin").text))
            ymin = round((float)(bnd_xml.find("ymin").text))
            xmax = round((float)(bnd_xml.find("xmax").text))
            ymax = round((float)(bnd_xml.find("ymax").text))
            class_id = self.class_dict[class_name]
            coords.append([xmin, ymin, xmax, ymax, class_id])

        coords.sort(key=lambda coord: (coord[2] - coord[0]) * (coord[3] - coord[1]))

        if self.is_train:

            transform_seed = random.randint(0, 3)

            if transform_seed == 0:  # 原图
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 1:  # 缩放+中心裁剪
                img, coords = image.center_crop_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 2:  # 平移
                img, coords = image.transplant_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            else:  # 曝光度调整
                img = image.exposure(img, gamma=0.5)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)
            # no !!
            '''
            else:  # 饱和度调整
                H, S, V = cv2.split(img)
                cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
                cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)
            '''
            # ==
        else:
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        ground_truth, ground_mask_positive, ground_mask_negative = self.getGroundTruth(coords)
        #print("ground_truth, ground_mask_positive, ground_mask_negative".format(ground_truth, ground_mask_positive, ground_mask_negative))
        return img, [ground_truth, ground_mask_positive, ground_mask_negative]
        #return img, [ground_truth, ground_mask_positive, ground_mask_negative, img_path]

'''
    def getGroundTruth(self, coords):

        feature_size = self.input_size // self.grid_size
        if self.loss_mode == "mse":
            ground_truth = np.zeros([feature_size, feature_size, 10 + self.class_num])
            #ground_mask_positive = np.zeros([feature_size, feature_size, 2 * 5 + self.class_num])
            #ground_mask_negative = np.zeros([feature_size, feature_size, 2 * 5 + self.class_num])

        else:
            ground_truth = np.zeros([feature_size, feature_size, 10 + 1])

        for coord in coords:
            # positive_num = positive_num + 1
            # bounding box归一化
            xmin, ymin, xmax, ymax, class_id = coord

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            index_row = (int)(center_y * feature_size)
            index_col = (int)(center_x * feature_size)

            # 分类标签 label_smooth
            if self.loss_mode == "mse":
                # 转化为one_hot编码 对one_hot编码做平滑处理
                class_list = np.full(shape=self.class_num, fill_value=1.0, dtype=np.float)
                deta = 0.01
                class_list = class_list * deta / (self.class_num - 1)
                class_list[class_id] = 1.0 - deta
            elif self.loss_mode == "cross_entropy":
                class_list = [class_id]
            else:
                raise Exception("the loss mode can't be support now!")
            # 计算中心点坐标(x,y) = ((x_max, y_max) + (x_min, y_min)) * 0.5
            # 计算宽高(w,h) = (x_max, y_max) - (x_min, y_min)
            # 拼接成一个数组(x, y, w, h)
            #bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            #class_one_hot = [0 for _ in range(self.class_num)]
            #class_list = [self.label_smooth_value / (self.class_num - 1) for i in range(self.class_num)]
            #class_list[class_index] = 1 - self.label_smooth_value

            # 定位数据预设
            ground_box = [center_x * feature_size - index_col, center_y * feature_size - index_row,
                          ground_width, ground_height, 1,
                          round(xmin * self.input_size), round(ymin * self.input_size),
                          round(xmax * self.input_size), round(ymax * self.input_size),
                          round(ground_width * self.input_size * ground_height * self.input_size)
                          ]
            ground_box.extend(class_list)
            #ground_box.extend()

            ground_truth[index_row][index_col] = np.array(ground_box)

        return ground_truth
'''

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
class voc_dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

import torch
class voc_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_label = next(self.loader)
        except  StopIteration:
            self.next_data = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.float().to(device=self.device,non_blocking=True)
            self.next_label[0] = self.next_label[0].float().to(device=self.device,non_blocking=True)
            self.next_label[1] = self.next_label[1].to(device=self.device, non_blocking=True)
            self.next_label[2] = self.next_label[2].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        labels = self.next_label
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if labels is not None:
            for label_idx in range(len(self.next_label)):
                self.next_label[label_idx].record_stream(torch.cuda.current_stream())

        self.preload()
        return data, labels

'''
device = torch.device("cuda:0")
dataset = VOC_Detection_Set()
img, label = dataset.test_read_data()
img = img.unsqueeze(0).to(device=device)
label[0] = torch.FloatTensor(label[0]).to(device=device)
label[1] = torch.BoolTensor(label[1]).to(device=device)
label[2] = torch.BoolTensor(label[2]).to(device=device)
print(label[1].int().sum())
print(label[2].int().sum())
from YOLO.Train.YOLOv1_LossFunction import YOLOv1_Loss
yolo_loss = YOLOv1_Loss().to(device=device)
from YOLO.Train.YOLOv1_Model import YOLOv1
yolo = YOLOv1().to(device=device)
loss = yolo_loss(bounding_boxes=yolo(img), ground_labels=label)
print(loss)
'''
