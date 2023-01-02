#------0.common variable definition------

import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from utils.model import accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.model import feature_map_visualize
from YOLO.PreTrain.YOLO_Feature import YOLO_Feature
from DataSet.COCO_DataSet import coco_classify_dataset
from DataSet.ImageNet_DataSet import ImageNet

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

if __name__ == "__main__":

    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLO_Feature train config")
    parser.add_argument('--batch_size', type=int, help="YOLO_Feature train batch_size", default=64)
    parser.add_argument('--num_workers', type=int, help="YOLO_Feature train num_worker num", default=4)
    parser.add_argument('--lr', type=float, help="lr", default=3e-4)
    parser.add_argument('--weight_decay', type=float, help="weight_decay", default=0.0005)
    parser.add_argument('--epoch_num', type=int, help="YOLO_Feature train epoch_num", default=200)
    parser.add_argument('--epoch_interval', type=int, help="save YOLO_Feature interval", default=10)
    parser.add_argument('--class_num', type=int, help="YOLO_Feature train class_num", default=80)
    parser.add_argument('--data_category', type=str, help="YOLO_Feature data category", default="COCO", choices=["COCO", "ImageNet"])
    parser.add_argument('--train_imgs', type=str, help="YOLO_Feature train train_imgs", default="../../DataSet/COCO2017/Train/Imgs")
    parser.add_argument('--train_labels', type=str, help="YOLO_Feature train train_labels", default="../../DataSet/COCO2017/Train/Labels")
    parser.add_argument('--val_imgs', type=str, help="YOLO_Feature train val_imgs", default="../../DataSet/COCO2017/Val/Imgs")
    parser.add_argument('--val_labels', type=str, help="YOLO_Feature train val_labels", default="../../DataSet/COCO2017/Val/Labels")
    parser.add_argument('--grad_visualize', type=bool, help="YOLO_Feature train grad visualize", default=False)
    parser.add_argument('--feature_map_visualize', type=bool, help="YOLO_Feature train feature map visualize", default=False)
    parser.add_argument('--restart', type=bool, help="YOLO_Feature train from zeor?", default=True)
    parser.add_argument('--pre_weight_file', type=str, help="YOLO_Feature pre weight path", default="./weights/YOLO_Feature_20.pth")
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    epoch_num = args.epoch_num
    epoch_interval = args.epoch_interval
    class_num = args.class_num

    if args.restart == True:
        lr = args.lr
        param_dict = {}
        epoch = 0
        epoch_val_loss_min = 999999999

    else:
        param_dict = torch.load(args.pre_weight_file, map_location=torch.device("cpu"))
        optimal_dict = param_dict['optimal']
        epoch = param_dict['epoch']
        epoch_val_loss_min = param_dict['epoch_val_loss_min']

    # 2.dataset
    if args.data_category == "COCO":
        train_dataSet = coco_classify_dataset(imgs_path=args.train_imgs,txts_path=args.train_labels, is_train=True, edge_threshold=200)
        val_dataSet = coco_classify_dataset(imgs_path=args.val_imgs,txts_path=args.val_labels, is_train=False, edge_threshold=200)
    else:
        from torchvision.transforms import transforms
        imagenet_transform = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化后.不容易产生梯度爆炸的问题
        ])
        train_dataSet = ImageNet(root=args.train_imgs, transform=imagenet_transform)
        val_dataSet = ImageNet(root=args.val_imgs, transform=imagenet_transform)

    # 3-4.network - optimizer
    yolo_feature = YOLO_Feature(classes_num=class_num)
    if args.restart == True:
        yolo_feature.initialize_weights()
        optimizer = optim.Adam(params=yolo_feature.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        yolo_feature.load_state_dict(param_dict['model'])
        optimizer = param_dict['optimizer']
    yolo_feature.to(device=device, non_blocking=True)

    # 5.loss
    loss_function = nn.CrossEntropyLoss().to(device=device)

    # 6.train and record
    input_size = 256
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

    while epoch < epoch_num:

        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_top1_acc = 0
        epoch_train_top5_acc = 0
        epoch_val_top1_acc = 0
        epoch_val_top5_acc = 0

        train_loader = DataLoader(dataset=train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        train_len = train_loader.__len__()
        yolo_feature.train()
        with tqdm(total=train_len) as tbar:

            for batch_index, batch_train in enumerate(train_loader):
                train_data = batch_train[0].float().to(device=device, non_blocking=True)
                label_data = batch_train[1].long().to(device=device, non_blocking=True)
                net_out = yolo_feature(train_data)
                loss = loss_function(net_out, label_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                # 计算准确率
                net_out = net_out.detach()
                [top1_acc, top5_acc] = accuracy(net_out, label_data)
                top1_acc = top1_acc.item()
                top5_acc = top5_acc.item()

                epoch_train_top1_acc = epoch_train_top1_acc + top1_acc
                epoch_train_top5_acc = epoch_train_top5_acc + top5_acc

                tbar.set_description(
                    "train: class_loss:{} top1-acc:{} top5-acc:{}".format(round(loss.item(), 4), round(top1_acc, 4),
                                                                          round(top5_acc, 4), refresh=True))
                tbar.update(1)

                if args.feature_map_visualize:
                    feature_map_visualize(train_data[0][0], writer, yolo_feature)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print(
                "train-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_train_loss / train_loader.__len__(), 4), round(
                    epoch_train_top1_acc / train_loader.__len__(), 4), round(
                    epoch_train_top5_acc / train_loader.__len__(), 4)))

        # lr_reduce_scheduler.step()

        val_loader = DataLoader(dataset=val_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True)
        val_len = val_loader.__len__()
        yolo_feature.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_index, batch_train in enumerate(val_loader):
                    train_data = batch_train[0].float().to(device=device, non_blocking=True)
                    label_data = batch_train[1].long().to(device=device, non_blocking=True)
                    net_out = yolo_feature(train_data)
                    loss = loss_function(net_out, label_data)
                    batch_loss = loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    # 计算准确率
                    net_out = net_out.detach()
                    [top1_acc, top5_acc] = accuracy(net_out, label_data)
                    top1_acc = top1_acc.item()
                    top5_acc = top5_acc.item()

                    epoch_val_top1_acc = epoch_val_top1_acc + top1_acc
                    epoch_val_top5_acc = epoch_val_top5_acc + top5_acc

                    tbar.set_description(
                        "val: class_loss:{} top1-acc:{} top5-acc:{}".format(round(loss.item(), 4), round(top1_acc, 4),
                                                                            round(top5_acc, 4), refresh=True))
                    tbar.update(1)

                if args.feature_map_visualize:
                    feature_map_visualize(train_data[0][0], writer, yolo_feature)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print(
                "val-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_val_loss / val_loader.__len__(), 4), round(
                    epoch_val_top1_acc / val_loader.__len__(), 4), round(
                    epoch_val_top5_acc / val_loader.__len__(), 4)))
        epoch = epoch + 1

        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = yolo_feature.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = yolo_feature.state_dict()
            param_dict['optimizer'] = optimizer
            param_dict['epoch'] = epoch
            param_dict['optimal'] = optimal_dict
            param_dict['epoch_val_loss_min'] = epoch_val_loss_min
            torch.save(param_dict, './weights/YOLO_Feature_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

        avg_train_sample_loss = epoch_train_loss / batch_size / train_loader.__len__()
        avg_val_sample_loss = epoch_val_loss / batch_size / val_loader.__len__()

        print("epoch:{}, train_sample_avg_loss:{}, val_sample_avg_loss:{}".format(epoch, avg_train_sample_loss, avg_val_sample_loss))

        if args.grad_visualize:
            for i, (name, layer) in enumerate(yolo_feature.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch)

        writer.add_scalar('Train/Loss_sample', avg_train_sample_loss, epoch)
        writer.add_scalar('Train/Batch_Acc_Top1', round(epoch_train_top1_acc / train_loader.__len__(), 4), epoch)
        writer.add_scalar('Train/Batch_Acc_Top5', round(epoch_train_top5_acc / train_loader.__len__(), 4), epoch)

        writer.add_scalar('Val/Loss_sample', avg_val_sample_loss, epoch)
        writer.add_scalar('Val/Batch_Acc_Top1', round(epoch_val_top1_acc / val_loader.__len__(), 4), epoch)
        writer.add_scalar('Val/Batch_Acc_Top5', round(epoch_val_top5_acc / val_loader.__len__(), 4), epoch)

    writer.close()