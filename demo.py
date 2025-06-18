#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
import statistics
from time import perf_counter
import os
import numpy as np
import openvino as ov
from openvino.utils.types import get_dtype
import torch
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.data.datasets import MVTecADDataset
import torchvision.transforms as transforms
import time
import argparse
from pathlib import Path
import cv2
import random

mvtec_categorys = {
    1: 'bottle', 
    2: 'cable', 
    3: 'capsule', 
    4: 'carpet', 
    5: 'grid', 
    6: 'hazelnut', 
    7: 'leather', 
    8: 'metal_nut', 
    9: 'pill', 
    10: 'screw', 
    11: 'tile',  
    12: 'toothbrush',  
    13: 'transistor',  
    14: 'wood', 
    15: 'zipper'
}

def arg_parser():
    parser = argparse.ArgumentParser(description='Performance test for model inference pipeline', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', type=str, default="ovmodels/patchcore_resnet18_bottle/model.xml", help='the OpenVINO IR *.xml')
    parser.add_argument('-d', '--device', default='CPU', help=f'The device to do inference')
    parser.add_argument('-cn', '--class_name', nargs='+', default=['bottle'], help=f'The class name of mvtec, default is [bottle]\n {mvtec_categorys}\n(default: %(default)s)')

    return parser.parse_args()

def completion_callback(infer_request: ov.InferRequest, data) -> None:
    pred_score = infer_request.get_tensor("pred_score").data
    anomaly_map = infer_request.get_tensor("anomaly_map").data
    pred_mask = infer_request.get_tensor("pred_mask").data


def save_results_callback(infer_request: ov.InferRequest, data) -> None:
    anomaly_map = infer_request.get_tensor("anomaly_map").data
    pred_mask = infer_request.get_tensor("pred_mask").data
    data.anomaly_map=torch.tensor(anomaly_map.squeeze())
    data.pred_mask=torch.tensor(pred_mask.squeeze())

    result_map = visualize_image_item(
        data,
        fields=["image", "anomaly_map"],
        fields_config={
            "anomaly_map": {"colormap": True, "normalize": True}
        }
    )
    base_img_path = Path("_".join(data.image_path.split('/')[-4:]))
    amap_path = "res/anomaly_map/"+ base_img_path.name
    overlay_path = "res/overlay/" + base_img_path.name
    result_map.save(amap_path)
    result_overlay = visualize_image_item(
        data,
        overlay_fields=[("image", ["gt_mask", "pred_mask"])],
        overlay_fields_config={
            "gt_mask": {"mode": "contour", "color": (0, 255, 0), "alpha": 0.7},
            "pred_mask": {"mode": "contour", "color": (255, 0, 0), "alpha": 0.7}
        }
    )
    result_overlay.save(overlay_path)


def resize_with_padding(img, target_size, pad_color=(0, 0, 0), centering=True):
    """
    保持宽高比缩放图像，并通过填充适配目标尺寸
    
    参数：
    img: 输入图像（HWC格式，numpy数组）
    target_size: 目标尺寸 (width, height)
    pad_color: 填充颜色（BGR格式，默认黑色）
    centering: 是否居中填充（默认True）
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放后的尺寸
    if w * target_h > h * target_w:
        new_w = target_w
        new_h = int(h * target_w / w)
    else:
        new_h = target_h
        new_w = int(w * target_h / h)
    
    # 缩放图像
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 计算填充尺寸
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    if centering:
        # 居中填充（上下左右对称）
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
    else:
        # 居左/上填充（右侧/下侧填充）
        top = 0
        bottom = pad_h
        left = 0
        right = pad_w
    
    # 添加填充
    img_padded = cv2.copyMakeBorder(
        img_resized,
        top=top, bottom=bottom,
        left=left, right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    
    return img_padded



def main():
    args = arg_parser()
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {ov.__version__}")

    device_name = args.device
    
    os.makedirs("res/anomaly_map", exist_ok=True)
    os.makedirs("res/overlay", exist_ok=True)


    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}

    total_latency = []
    total_tput = []

    print("Please select the category for demo：")
    for key, value in mvtec_categorys.items():
        print(f"{key}. {value}")
    choice = int(input("Entry the number："))

    if choice in mvtec_categorys:
        class_name = mvtec_categorys[choice]
        print(f"You select the category '{class_name}' to run the demo")
    else:
        print("Invalid Input!")

    model_path = f'ovmodels/patchcore_resnet18_{class_name}/model.xml'
    if not os.path.exists(model_path):
        raise ValueError(f"Model file does not exist at {model_path}, Please export the OpenVINO IR for {class_name}, or download from https://github.com/junx8/ov_pipeline_benchmark/releases/download/v1.4/ovmodels.tar.gz")
    core = ov.Core()
    inmodel = core.read_model(model_path)
    inmodel.reshape([1,3,256,256])
    compiled_model = core.compile_model(inmodel, device_name, tput)

    dataset = MVTecADDataset(
        root="./datasets/MVTecAD",
        category=class_name,
        split="test"
    )
    from screeninfo import get_monitors

    # 获取主显示器分辨率
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height

    # 创建全屏窗口并调整大小（可选）
    cv2.namedWindow("x", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("x", screen_width, screen_height)
    cv2.setWindowProperty("x", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    label_name = {0: "Normal",
                  1: "Abnomal"}
    data_num = len(dataset)
    while True:
        index = random.randint(0, data_num-1)
        data = dataset[index]
        stime = time.time()
        resize_transform = transforms.Resize((256, 256))
        resize_img = resize_transform(data.image)
        nimg = np.expand_dims(resize_img, axis=0)
        input_tensor = ov.Tensor(array=nimg, shared_memory=True)

        results = compiled_model.infer_new_request({0: input_tensor})
        dtime = time.time() - stime

        pred_mask = results['pred_mask']
        pred_score = results['pred_score']
        anomaly_map = results['anomaly_map']
        pred_label = label_name[results['pred_label'][0]]

        data.anomaly_map=torch.tensor(anomaly_map.squeeze())
        data.pred_mask=torch.tensor(pred_mask.squeeze())

        result_overlay = visualize_image_item(
            data,
            overlay_fields=[("image", ["pred_mask"])],
            overlay_fields_config={
                "pred_mask": {"mode": "fill", "color": (255, 0, 0), "alpha": 0.5}
            },
            text_config={"enable": False}
        )
        show_img_0 = (data.image.numpy().transpose(1,2,0)*255).astype(np.uint8)[:, :, [2, 1, 0]]
        edge = np.zeros_like(show_img_0)[:,0:10,:]
        show_img_1 = cv2.resize(np.array(result_overlay)[:, :, [2, 1, 0]], show_img_0.shape[0:2])
        show_img = np.hstack([show_img_0, edge, show_img_1])
        x_s = int(screen_width/2 + 100)


        cv2.putText(
            show_img, 
            f"GT Label: {label_name[int(data.gt_label)]}", 
            org=(50, 50), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.0, 
            color=(0, 255, 0), 
            thickness=2
        )
        
        cv2.putText(
            show_img, 
            f"Pred Label: {pred_label}", 
            org=(x_s, 50), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.0, 
            color=(255, 0, 0), 
            thickness=2
        )
        cv2.putText(
            show_img, 
            f"Pred Score: {float(pred_score):.3f}", 
            org=(x_s, 100), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.0, 
            color=(255, 0, 0), 
            thickness=2
        )
        cv2.putText(
            show_img, 
            f"Latency: {dtime*1000:.0f} ms", 
            org=(x_s, 150), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.0, 
            color=(255, 0, 0), 
            thickness=2
        )
        # import ipdb;ipdb.set_trace()
        rs_show_img = resize_with_padding(show_img, (screen_width, screen_height), pad_color=(255, 255, 255))
        cv2.imshow('x',rs_show_img)
        key = cv2.waitKey(1000)
        if key == 27:
            print('Detecting the "Esc" key, stopping!')
            cv2.destroyAllWindows()
            sys.exit()


if __name__ == '__main__':
    main()