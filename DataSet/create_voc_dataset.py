import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create voc label")
    parser.add_argument('--voc_pth', type=str, default='../../VOCdevkit/VOC2012')
    parser.add_argument('--train_ratio', type=float, default='0.9')
    parser.add_argument('--target_train_pth', type=str, default='./VOC2007+2012/Train')
    parser.add_argument('--target_val_pth', type=str, default='./VOC2007+2012/Val')
    args = parser.parse_args()

    voc_pth = args.voc_pth
    imgs_pth = os.path.join(voc_pth, "JPEGImages")
    annotations_pth = os.path.join(voc_pth, "Annotations")
    imgs_name = os.listdir(imgs_pth)

    train_samples_num = round(len(imgs_name) * args.train_ratio)

    train_imgs_name = imgs_name[ : train_samples_num]
    val_imgs_name = imgs_name[train_samples_num : ]

    for img_name in train_imgs_name:
        img_pth = os.path.join(imgs_pth, img_name)
        annotation_pth = os.path.join(annotations_pth, img_name.replace(".jpg", ".xml"))

        shutil.move(img_pth, os.path.join(args.target_train_pth, "JPEGImages"))
        shutil.move(annotation_pth, os.path.join(args.target_train_pth, "Annotations"))

    for img_name in val_imgs_name:
        img_pth = os.path.join(imgs_pth, img_name)
        annotation_pth = os.path.join(annotations_pth, img_name.replace(".jpg", ".xml"))

        shutil.move(img_pth, os.path.join(args.target_val_pth, "JPEGImages"))
        shutil.move(annotation_pth, os.path.join(args.target_val_pth, "Annotations"))
