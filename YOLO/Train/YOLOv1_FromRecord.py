#---------------step0:Common Definition-----------------
'''
import torch
import argparse
from utils.model import feature_map_visualize
from torch.utils.data import DataLoader
from DataSet.VOC_DataSet import VOC_Detection_Set
from YOLO.Train.YOLOv1_Model import YOLOv1
from YOLO.Train.YOLOv1_LossFunction import YOLOv1_Loss
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import model

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

if __name__ == "__main__":

    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLOv1 train config")
    parser.add_argument('--num_workers', type=int, help="train num_workers num", default=4)
    parser.add_argument('--B', type=int, help="YOLOv1 predict box num every grid", default=32)
    parser.add_argument('--class_num', type=int, help="YOLOv1 predict class num", default=20)
    parser.add_argument('--lr', type=float, help="start lr", default=0.001)
    parser.add_argument('--lr_mul_factor_epoch_1', type=float, help="lr mul factor when full YOLOv1 train in epoch1",
                        default=1.04)
    parser.add_argument('--lr_epoch_2', type=int, help="lr when full YOLOv1 train in epoch2", default=0.01)
    parser.add_argument('--lr_epoch_77', type=int, help="lr when full YOLOv1 train in epoch77", default=0.001)
    parser.add_argument('--lr_epoch_107', type=int, help="DarkNet53 train class_num", default=0.0001)
    parser.add_argument('--batch_size', type=int, help="YOLOv1 train batch size", default=24)
    parser.add_argument('--weight_decay', type=float, help="optim weight_decay", default=5e-4)
    parser.add_argument('--momentum', type=float, help="optim momentum", default=0.9)
    parser.add_argument('--weight_file', type=str, help="DarkNet53 pre-train path",
                        default="./weights/YOLO_V1_1.pth")
    parser.add_argument('--train_imgs', type=str, help="YOLOv1 train train_imgs",
                        default="../../DataSet/VOC2007+2012/Train/JPEGImages")
    parser.add_argument('--train_labels', type=str, help="YOLOv1 train train_labels",
                        default="../../DataSet/VOC2007+2012/Train/Annotations")
    parser.add_argument('--val_imgs', type=str, help="YOLOv1 train val_imgs",
                        default="../../DataSet/VOC2007+2012/Val/JPEGImages")
    parser.add_argument('--val_labels', type=str, help="YOLOv1 train val_labels",
                        default="../../DataSet/VOC2007+2012/Val/Annotations")
    parser.add_argument('--voc_classes_path', type=str, help="voc classes path",
                        default="../../DataSet/VOC2007+2012/class.data")
    parser.add_argument('--epoch_interval', type=int, help="save YOLOv1 weight epoch interval", default=1)
    parser.add_argument('--epoch_unfreeze', type=int, help="YOLOv1 backbone unfreeze epoch", default=False)
    parser.add_argument('--epoch_num', type=int, help="YOLOv1 train epoch num", default=160)
    parser.add_argument('--grad_visualize', type=bool, help="YOLOv1 train grad visualize", default=False)
    parser.add_argument('--feature_map_visualize', type=bool, help="YOLOv1 train feature map visualize", default=False)
    args = parser.parse_args()

    num_workers = args.num_workers
    class_num = args.class_num
    batch_size = args.batch_size
    lr = args.lr
    lr_mul_factor_epoch_1 = args.lr_mul_factor_epoch_1
    lr_epoch_2 = args.lr_epoch_2
    lr_epoch_77 = args.lr_epoch_77
    lr_epoch_107 = args.lr_epoch_107
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    momentum = args.momentum
    weight_file = args.weight_file
    param_dict = torch.load(weight_file, map_location=torch.device("cpu"))
    epoch_interval = args.epoch_interval
    epoch_unfreeze = args.epoch_unfreeze
    epoch_val_loss_min = 999999999
    loss_mode = "mse"

    # 2.dataset
    train_dataSet = VOC_Detection_Set(imgs_path=args.train_imgs,
                                      annotations_path=args.train_labels,
                                      classes_file=args.voc_classes_path, class_num=class_num, is_train=True,
                                      loss_mode=loss_mode)
    val_dataSet = VOC_Detection_Set(imgs_path=args.val_imgs,
                                    annotations_path=args.val_labels,
                                    classes_file=args.voc_classes_path, class_num=class_num, is_train=False,
                                    loss_mode=loss_mode)

    # 3.network
    YOLO = YOLOv1().to(device=device, non_blocking=True)
    YOLO.load_state_dict(param_dict['model'])
    epoch = param_dict['epoch']
    if epoch < epoch_unfreeze:
        model.set_freeze_by_idxs(YOLO, [0, 1])

    # 4.optimzer
    optimizer_SGD = param_dict['optim']

    # 5.loss
    loss_function = YOLOv1_Loss().to(device=device, non_blocking=True)

    # 6.train and record
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    while epoch <= args.epoch_num:

        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_iou = 0
        epoch_val_iou = 0
        epoch_train_object_num = 0
        epoch_val_object_num = 0
        epoch_train_loss_coord = 0
        epoch_val_loss_coord = 0
        epoch_train_loss_confidence = 0
        epoch_val_loss_confidence = 0
        epoch_train_loss_classes = 0
        epoch_val_loss_classes = 0

        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:

            for batch_index, batch_train in enumerate(train_loader):

                train_data = batch_train[0].float().to(device=device, non_blocking=True)
                label_data = batch_train[1].float().to(device=device, non_blocking=True)
                loss = loss_function(bounding_boxes=YOLO(train_data),ground_truth=label_data)
                sample_avg_loss = loss[0]
                epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
                epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
                epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
                epoch_train_iou = epoch_train_iou + loss[4]
                epoch_train_object_num = epoch_train_object_num + loss[5]
                sample_avg_loss.backward()
                optimizer_SGD.step()
                optimizer_SGD.zero_grad()
                batch_loss = sample_avg_loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                tbar.set_description("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                tbar.update(1)

                if epoch == epoch_unfreeze + 1:
                    lr = min(lr * lr_mul_factor_epoch_1, lr_epoch_2)
                    for param_group in optimizer_SGD.param_groups:
                        param_group["lr"] = lr

            print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4), round(epoch_train_loss_confidence / train_len, 4), round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou / epoch_train_object_num, 4)))

        val_len = val_loader.__len__()
        YOLO.eval()
        with torch.no_grad():
            with tqdm(total=val_len) as tbar:

                for batch_index, batch_train in enumerate(val_loader):
                    train_data = batch_train[0].float().cuda(device=0)
                    label_data = batch_train[1].float().cuda(device=0)
                    loss = loss_function(bounding_boxes=YOLO(train_data), ground_truth=label_data)
                    sample_avg_loss = loss[0]
                    epoch_val_loss_coord = epoch_val_loss_coord + loss[1]
                    epoch_val_loss_confidence = epoch_val_loss_confidence + loss[2]
                    epoch_val_loss_classes = epoch_val_loss_classes + loss[3]
                    epoch_val_iou = epoch_val_iou + loss[4]
                    epoch_val_object_num = epoch_val_object_num + loss[5]
                    batch_loss = sample_avg_loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    tbar.set_description("val: coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                    tbar.update(1)

                if args.feature_map_visualize:
                    feature_map_visualize(train_data[0][0], writer, YOLO)
                    # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
                print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4), round(epoch_val_loss_confidence / val_len, 4), round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou / epoch_val_object_num, 4)))

        epoch = epoch + 1

        if epoch == epoch_unfreeze:
            model.unfreeze_by_idxs(YOLO, [0, 1])

        if epoch == 2 + epoch_unfreeze:
            lr = lr_epoch_2
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
        elif epoch == 77 + epoch_unfreeze:
            lr = lr_epoch_77
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
        elif epoch == 107 + epoch_unfreeze:
            lr = lr_epoch_107
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr

        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = YOLO.state_dict()

        if epoch % epoch_interval == 0:
            param_dict["model"] = YOLO.state_dict()
            param_dict["optim"] = optimizer_SGD
            param_dict["epoch"] = epoch
            torch.save(param_dict, './weights/YOLO_V1_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log',filename_suffix='[' + str(epoch) + '~' + str(epoch +00)+']')
        print("epoch : {} ; loss : {}".format(epoch,{epoch_train_loss}))

        if args.grad_visualize:
            for i, (name, layer) in enumerate(YOLO.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch)


        #for name, layer in YOLO.named_parameters():
            #writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            #writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
        writer.add_scalar('Train/Loss_confidenct', epoch_train_loss_confidence, epoch)
        writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
        writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)

    writer.close()
'''
#---------------step0:Common Definition-----------------
import torch
import argparse
from utils.model import feature_map_visualize
from DataSet.VOC_DataSet import VOC_Detection_Set
from torch.utils.data import DataLoader
from YOLO.Train.YOLOv1_Model import YOLOv1
from utils import model
from YOLO.Train.YOLOv1_LossFunction import YOLOv1_Loss
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLOv1 train config")
    parser.add_argument('--num_workers', type=int, help="train num_workers num", default=4)
    parser.add_argument('--B', type=int, help="YOLOv1 predict box num every grid", default=32)
    parser.add_argument('--class_num', type=int, help="YOLOv1 predict class num", default=20)
    parser.add_argument('--lr', type=float, help="start lr", default=0.001)
    parser.add_argument('--lr_mul_factor_epoch_1', type=float, help="lr mul factor when full YOLOv1 train in epoch1",
                        default=1.04)
    parser.add_argument('--lr_epoch_2', type=int, help="lr when full YOLOv1 train in epoch2", default=0.01)
    parser.add_argument('--lr_epoch_77', type=int, help="lr when full YOLOv1 train in epoch77", default=0.001)
    parser.add_argument('--lr_epoch_107', type=int, help="DarkNet53 train class_num", default=0.0001)
    parser.add_argument('--batch_size', type=int, help="YOLOv1 train batch size", default=32)
    parser.add_argument('--weight_decay', type=float, help="optim weight_decay", default=5e-4)
    parser.add_argument('--momentum', type=float, help="optim momentum", default=0.9)
    parser.add_argument('--weight_file', type=str, help="YOLOv1 weight path",
                        default="./weights/YOLO_V1_110.pth")
    parser.add_argument('--train_imgs', type=str, help="YOLOv1 train train_imgs",
                        default="../../DataSet/VOC2007+2012/Train/JPEGImages")
    parser.add_argument('--train_labels', type=str, help="YOLOv1 train train_labels",
                        default="../../DataSet/VOC2007+2012/Train/Annotations")
    parser.add_argument('--val_imgs', type=str, help="YOLOv1 train val_imgs",
                        default="../../DataSet/VOC2007+2012/Val/JPEGImages")
    parser.add_argument('--val_labels', type=str, help="YOLOv1 train val_labels",
                        default="../../DataSet/VOC2007+2012/Val/Annotations")
    parser.add_argument('--voc_classes_path', type=str, help="voc classes path",
                        default="../../DataSet/VOC2007+2012/class.data")
    parser.add_argument('--epoch_interval', type=int, help="save YOLOv1 weight epoch interval", default=10)
    parser.add_argument('--epoch_unfreeze', type=int, help="YOLOv1 backbone unfreeze epoch", default=False)
    parser.add_argument('--epoch_num', type=int, help="YOLOv1 train epoch num", default=160)
    parser.add_argument('--grad_visualize', type=bool, help="YOLOv1 train grad visualize", default=False)
    parser.add_argument('--feature_map_visualize', type=bool, help="YOLOv1 train feature map visualize", default=False)
    args = parser.parse_args()

    num_workers = args.num_workers
    class_num = args.class_num
    batch_size = args.batch_size
    lr = args.lr
    lr_mul_factor_epoch_1 = args.lr_mul_factor_epoch_1
    lr_epoch_2 = args.lr_epoch_2
    lr_epoch_77 = args.lr_epoch_77
    lr_epoch_107 = args.lr_epoch_107
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    momentum = args.momentum
    weight_file = args.weight_file
    param_dict = torch.load(weight_file, map_location=torch.device("cpu"))
    epoch_interval = args.epoch_interval
    epoch_unfreeze = args.epoch_unfreeze
    epoch_val_loss_min = param_dict['epoch_val_loss_min']
    loss_mode = "mse"

    # 2.dataset
    train_dataSet = VOC_Detection_Set(imgs_path=args.train_imgs,
                                      annotations_path=args.train_labels,
                                      classes_file=args.voc_classes_path, class_num=class_num, is_train=True,
                                      loss_mode=loss_mode)
    val_dataSet = VOC_Detection_Set(imgs_path=args.val_imgs,
                                    annotations_path=args.val_labels,
                                    classes_file=args.voc_classes_path, class_num=class_num, is_train=False,
                                    loss_mode=loss_mode)

    # 3.network
    YOLO = YOLOv1().to(device=device, non_blocking=True)
    YOLO.load_state_dict(param_dict['model'])
    epoch = param_dict['epoch']
    if epoch < epoch_unfreeze:
        model.set_freeze_by_idxs(YOLO, [0, 1])
        
    # 4.optimzer
    optimizer_SGD = optim.SGD(YOLO.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum)

    # 5.loss
    loss_function = YOLOv1_Loss().to(device=device, non_blocking=True)

    # 6.train and record
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
    param_dict = {}
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    while epoch < args.epoch_num:

        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_iou = 0
        epoch_val_iou = 0
        epoch_train_object_num = 0
        epoch_val_object_num = 0
        epoch_train_loss_coord = 0
        epoch_val_loss_coord = 0
        epoch_train_loss_pos_conf = 0
        epoch_train_loss_neg_conf = 0
        epoch_val_loss_pos_conf = 0
        epoch_val_loss_neg_conf = 0
        epoch_val_loss_confidence = 0
        epoch_train_loss_classes = 0
        epoch_val_loss_classes = 0

        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:

            for batch_idx, [train_data, label_data] in enumerate(train_loader):
                optimizer_SGD.zero_grad()
                train_data = train_data.float().to(device=device, non_blocking=True)
                label_data[0] = label_data[0].float().to(device=device, non_blocking=True)
                label_data[1] = label_data[1].to(device=device, non_blocking=True)
                label_data[2] = label_data[2].to(device=device, non_blocking=True)

                loss = loss_function(bounding_boxes=YOLO(train_data), ground_labels=label_data)
                sample_avg_loss = loss[0]
                epoch_train_loss_coord = epoch_train_loss_coord + loss[1] * batch_size
                epoch_train_loss_pos_conf = epoch_train_loss_pos_conf + loss[2] * batch_size
                epoch_train_loss_neg_conf = epoch_train_loss_neg_conf + loss[3] * batch_size
                epoch_train_loss_classes = epoch_train_loss_classes + loss[4] * batch_size
                epoch_train_iou = epoch_train_iou + loss[5]
                epoch_train_object_num = epoch_train_object_num + loss[6]

                sample_avg_loss.backward()
                optimizer_SGD.step()

                batch_loss = sample_avg_loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                tbar.set_description(
                    "train: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} avg_iou:{}".format(
                        round(loss[1], 4),
                        round(loss[2], 4),
                        round(loss[3], 4),
                        round(loss[4], 4),
                        round(loss[5] / loss[6],
                              4)), refresh=True)
                tbar.update(1)

                if epoch == epoch_unfreeze + 1:
                    lr = min(lr * lr_mul_factor_epoch_1, lr_epoch_2)
                    for param_group in optimizer_SGD.param_groups:
                        param_group["lr"] = lr

            if args.feature_map_visualize:
                feature_map_visualize(train_data[0][0], writer, YOLO)
            # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))

            print(
                "train-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                    round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4),
                    round(epoch_train_loss_pos_conf / train_len, 4), round(epoch_train_loss_neg_conf / train_len, 4),
                    round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou / epoch_train_object_num, 4)))

        val_len = val_loader.__len__()
        YOLO.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_idx, [train_data, label_data] in enumerate(val_loader):
                    train_data = train_data.float().to(device=device, non_blocking=True)
                    label_data[0] = label_data[0].float().to(device=device, non_blocking=True)
                    label_data[1] = label_data[1].to(device=device, non_blocking=True)
                    label_data[2] = label_data[2].to(device=device, non_blocking=True)
                    loss = loss_function(bounding_boxes=YOLO(train_data), ground_labels=label_data)
                    sample_avg_loss = loss[0]
                    epoch_val_loss_coord = epoch_val_loss_coord + loss[1] * batch_size
                    epoch_val_loss_pos_conf = epoch_val_loss_pos_conf + loss[2] * batch_size
                    epoch_val_loss_neg_conf = epoch_val_loss_neg_conf + loss[3] * batch_size
                    epoch_val_loss_classes = epoch_val_loss_classes + loss[4] * batch_size
                    epoch_val_iou = epoch_val_iou + loss[5]
                    epoch_val_object_num = epoch_val_object_num + loss[6]
                    batch_loss = sample_avg_loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    tbar.set_description(
                        "val: coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                            round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4], 4),
                            round(loss[5] / loss[6])), refresh=True)
                    tbar.update(1)

                if args.feature_map_visualize:
                    feature_map_visualize(train_data[0][0], writer, YOLO)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print("val-batch-mean loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} class_loss:{} iou:{}".format(
                round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4),
                round(epoch_val_loss_pos_conf / val_len, 4), round(epoch_val_loss_neg_conf / val_len, 4),
                round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou / epoch_val_object_num, 4)))

        epoch = epoch + 1

        if epoch == epoch_unfreeze:
            model.unfreeze_by_idxs(YOLO, [0, 1])

        if epoch == 2 + epoch_unfreeze:
            lr = lr_epoch_2
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
        elif epoch == 77 + epoch_unfreeze:
            lr = lr_epoch_77
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr
        elif epoch == 107 + epoch_unfreeze:
            lr = lr_epoch_107
            for param_group in optimizer_SGD.param_groups:
                param_group["lr"] = lr

        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = YOLO.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = YOLO.state_dict()
            param_dict['optim'] = optimizer_SGD
            param_dict['epoch'] = epoch
            param_dict['optimal'] = optimal_dict
            param_dict['epoch_val_loss_min'] = epoch_val_loss_min
            torch.save(param_dict, './weights/YOLO_V1_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
        print("epoch : {} ; loss : {}".format(epoch, {epoch_train_loss}))


        if args.grad_visualize:
            for i, (name, layer) in enumerate(YOLO.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch)
        
        #for name, layer in YOLO.named_parameters():
            #writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            #writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
        writer.add_scalar('Train/Loss_pos_conf', epoch_train_loss_pos_conf, epoch)
        writer.add_scalar('Train/Loss_neg_conf', epoch_train_loss_neg_conf, epoch)
        writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
        writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)

        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
        writer.add_scalar('Val/Loss_coord', epoch_val_loss_coord, epoch)
        writer.add_scalar('Val/Loss_pos_conf', epoch_val_loss_pos_conf, epoch)
        writer.add_scalar('Val/Loss_neg_conf', epoch_val_loss_neg_conf, epoch)
        writer.add_scalar('Val/Loss_classes', epoch_val_loss_classes, epoch)
        writer.add_scalar('Val/Epoch_iou', epoch_val_iou / epoch_val_object_num, epoch)

    writer.close()