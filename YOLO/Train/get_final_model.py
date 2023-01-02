# coding: utf-8
# use this file to get model weight
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./weights/YOLO_V1_160.pth")

    args = parser.parse_args()
    model_path = args.model_path

    param_dict = torch.load(model_path, map_location=torch.device("cpu"))
    torch.save(param_dict['model'], "./weights/YOLOv1_final.pth")
    torch.save(param_dict['optimal'], "./weights/YOLOv1_optimal.pth")
