#------0.common variable definition------
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from utils.model import accuracy, feature_map_visualize
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from YOLO.PreTrain.COCO_Classify_DataSet import coco_classify_dataset
from YOLO.PreTrain.YOLO_Feature import YOLO_Feature

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

if __name__ == "__main__":

    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLO_Feature train config")
    parser.add_argument('--param_pth', type=str, help="YOLO_Feature checkpoint pth", default="./weights/YOLO_Feature_140.pth")
    parser.add_argument('--batch_size', type=int, help="YOLO_Feature train batch_size", default=32)
    parser.add_argument('--num_workers', type=int, help="YOLO_Feature train num_worker num", default=4)
    parser.add_argument('--epoch_num', type=int, help="YOLO_Feature train epoch_num", default=200)
    parser.add_argument('--epoch_interval', type=int, help="save YOLO_Feature interval", default=10)
    parser.add_argument('--class_num', type=int, help="YOLO_Feature train class_num", default=80)
    parser.add_argument('--train_imgs', type=str, help="YOLO_Feature train train_imgs", default="../../DataSet/COCO2017/Train/Imgs")
    parser.add_argument('--train_labels', type=str, help="YOLO_Feature train train_labels", default="../../DataSet/COCO2017/Train/Labels")
    parser.add_argument('--val_imgs', type=str, help="YOLO_Feature train val_imgs", default="../../DataSet/COCO2017/Val/Imgs")
    parser.add_argument('--val_labels', type=str, help="YOLO_Feature train val_labels", default="../../DataSet/COCO2017/Val/Labels")
    parser.add_argument('--grad_visualize', type=bool, help="YOLO_Feature train grad visualize", default=False)
    parser.add_argument('--feature_map_visualize', type=bool, help="YOLO_Feature train feature map visualize", default=False)
    args = parser.parse_args()

    param_pth = args.param_pth
    param_dict = torch.load(param_pth, map_location=torch.device("cpu"))
    batch_size = args.batch_size
    num_workers = args.num_workers
    epoch_num = args.epoch_num
    epoch_interval = args.epoch_interval
    class_num = args.class_num
    epoch = param_dict['epoch']
    min_val_loss = param_dict['min_val_loss']

    # 2.dataset
    train_dataSet = coco_classify_dataset(imgs_path=args.train_imgs, txts_path=args.train_labels, is_train=True, edge_threshold=200)
    val_dataSet = coco_classify_dataset(imgs_path=args.val_imgs, txts_path=args.val_labels, is_train=False, edge_threshold=200)

    # 3.network
    yolo_feature = YOLO_Feature(classes_num=class_num)
    yolo_feature.load_state_dict(param_dict['model'])
    yolo_feature.to(device=device, non_blocking=True)

    # 4.optimzer
    optimizer = param_dict['optim']

    # 5.loss
    loss_function = nn.CrossEntropyLoss().to(device=device)

    # 6.train and record
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
                    "train: class_loss:{} top1-acc:{} top5-acc:{}".format(loss.item(), round(top1_acc, 4),
                                                                          round(top5_acc, 4), refresh=True))
                tbar.update(1)

            if args.feature_map_visualize:
                feature_map_visualize(train_data[0][0], writer, yolo_feature)
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
                        "val: class_loss:{} top1-acc:{} top5-acc:{}".format(loss.item(), round(top1_acc, 4),
                                                                            round(top5_acc, 4), refresh=True))
                    tbar.update(1)

                if args.feature_map_visualize:
                    feature_map_visualize(train_data[0][0], writer, yolo_feature)
            print(
                "train-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_val_loss / val_loader.__len__(), 4), round(
                    epoch_val_top1_acc / val_loader.__len__(), 4), round(
                    epoch_val_top5_acc / val_loader.__len__(), 4)))
        epoch = epoch + 1

        # save model weight which has the lowest val loss
        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            param_dict['min_val_loss'] = min_val_loss
            param_dict['min_loss_model'] = yolo_feature.state_dict()

        # save model as pth
        if epoch % epoch_interval == 0:
            param_dict['model'] = yolo_feature.state_dict()
            param_dict['optim'] = optimizer
            param_dict['epoch'] = epoch
            torch.save(param_dict, './weights/YOLO_Feature_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
        print("epoch : {} ; loss : {}".format(epoch, {epoch_train_loss}))

        if args.grad_visualize:
            for i, (name, layer) in enumerate(yolo_feature.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch)

        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
    writer.close()