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

mvtec_categorys = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',  'toothbrush',  'transistor',  'wood', 'zipper']

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


def main():
    args = arg_parser()
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {ov.__version__}")

    device_name = args.device
    
    os.makedirs("res/anomaly_map", exist_ok=True)
    os.makedirs("res/overlay", exist_ok=True)

    categorys = []
    for icls in args.class_name:
        if icls in mvtec_categorys:
            categorys.append(icls)

    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}

    total_latency = []
    total_tput = []
    for class_name in categorys:
        model_path = f'ovmodels/patchcore_resnet18_{class_name}/model.xml'
        core = ov.Core()
        inmodel = core.read_model(model_path)
        inmodel.reshape([1,3,256,256])
        compiled_model = core.compile_model(inmodel, device_name, tput)

        dataset = MVTecADDataset(
            root="./datasets/MVTecAD",
            category=class_name,
            split="test"
        )
        infer_queue = ov.AsyncInferQueue(compiled_model)
        infer_queue.set_callback(completion_callback)
        resize_transform = transforms.Resize((256, 256))

        print("\nStarting Performance Testing...")
        stime = time.time()
        for data in dataset:
            resize_img = resize_transform(data.image)
            nimg = np.expand_dims(resize_img, axis=0)
            input_tensor = ov.Tensor(array=nimg, shared_memory=True)
            infer_queue.start_async({0: input_tensor}, data)
        infer_queue.wait_all()

        dtime = time.time() - stime
        print(f'\nPerformance for {class_name}:\n - {len(dataset)} images task {dtime:.2f} sec, {dtime/len(dataset):.3f} sec/image')
        print(f' - throughput {len(dataset)/dtime:.2f} FPS')
        total_latency.append(dtime/len(dataset))
        total_tput.append(len(dataset)/dtime)
        print("\nSaving the results...")
        sinfer_queue = ov.AsyncInferQueue(compiled_model)
        sinfer_queue.set_callback(save_results_callback)

        stime = time.time()
        for data in dataset:
            resize_transform = transforms.Resize((256, 256))
            resize_img = resize_transform(data.image)
            nimg = np.expand_dims(resize_img, axis=0)
            input_tensor = ov.Tensor(array=nimg, shared_memory=True)
            sinfer_queue.start_async({0: input_tensor}, data)
        sinfer_queue.wait_all()

        dtime = time.time() - stime
        print(f'\nSaving Results Performance for {class_name}:\n - {len(dataset)} images task {dtime:.2f} sec, {dtime/len(dataset):.3f} sec/image')
        print(f' - throughput {len(dataset)/dtime:.2f} FPS')
    print("\n\n--------------------------------")
    print(f'\nAverage Performance for mvtec:\n')
    print(f' - Average Latency: {sum(total_latency)/len(total_latency):.3f} sec')
    print(f' - Average Throughput: {sum(total_tput)/len(total_tput):.2f} FPS\n')


if __name__ == '__main__':
    main()