# coding: utf-8
import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet DataSet")
    parser.add_argument('--source_pth', type=str, default='./ImageNet')
    parser.add_argument('--train_ratio', type=float, default='0.9')
    parser.add_argument('--train_pth', type=str, default='./ImageNet2012/Train')
    parser.add_argument('--val_pth', type=str, default='./ImageNet2012/Val')
    args = parser.parse_args()

    source_pth = args.source_pth
    train_ratio = args.train_ratio
    train_pth = args.train_pth
    val_pth = args.val_pth

    class_dir_list = os.listdir(source_pth)
    for class_dir in class_dir_list:
        class_pth = os.path.join(source_pth, class_dir)
        image_list = os.listdir(class_pth)
        idx_threshold = train_ratio * len(image_list)
        target_train_pth = os.path.join(train_pth, class_dir)
        target_val_pth = os.path.join(val_pth, class_dir)
        os.mkdir(target_train_pth)
        os.mkdir(target_val_pth)
        for img_idx in range(len(image_list)):
            if img_idx < idx_threshold:
                shutil.move(os.path.join(class_pth, image_list[img_idx]), target_train_pth)
            else:
                shutil.move(os.path.join(class_pth, image_list[img_idx]), target_val_pth)