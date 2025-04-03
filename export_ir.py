import argparse
from pathlib import Path
from anomalib.data import MVTec
from anomalib.models import WinClip,Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.deploy.export import CompressionType

mvtec_categorys = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',  'toothbrush',  'transistor',  'wood', 'zipper']

def arg_parser():
    parser = argparse.ArgumentParser(description='Export IR for WinCLIP + MvTec args parser', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ks', '--k_shot', type=int, default=0, help='k-shot: 1,2,3,4...\n(default: %(default)s)')
    parser.add_argument('-cn', '--class_name', nargs='+', default=['bottle'], help=f'The class name of mvtec, default is [bottle]\n {mvtec_categorys}\n(default: %(default)s)')
    parser.add_argument('-t', '--export_type', choices=['fp32', 'fp16','int8', 'int8_ptq', 'int8_acq'], default='fp16', help='Specify export IR Type: fp32/fp16/int8/int8_ptq/int8_acq\n(default: %(default)s)')
    parser.add_argument('-o', '--output', type=Path, default='ovmodels', required=False, help='Path to save the output OpenVINO IR.\n(default: %(default)s)')
    parser.add_argument('-et', '--export_threshold', action='store_true', help='Export Threshold for OpenVINO IR\n(default: %(default)s)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size for dataloader \nLarge size will OOM for INT8 PTQ, please set 4\n(default: %(default)s)')

    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    categorys = []
    for icls in args.class_name:
        if icls in mvtec_categorys:
            categorys.append(icls)

    for class_name in categorys:
        print(f"Starting export IR for {class_name} {args.k_shot}-shot...")

        # Initialize the datamodule, model and engine，default category is bottle
        # Default: Collecting reference images from training dataset
        datamodule = MVTec(category=class_name, train_batch_size=1, eval_batch_size=args.batch_size)
        model = Patchcore(backbone="resnet18")
        # datamodule = MVTec()
        # model = WinClip(k_shot=args.k_shot)
        engine = Engine()
        if args.export_threshold:
            engine.fit(datamodule=datamodule, model=model)
        else:
            model.model.setup(class_name=class_name)

        if args.export_type == "fp32":
            export_ir = engine.export(model=model, export_type=ExportType.OPENVINO, \
                                        export_root = args.output.joinpath(f"{class_name}/fp32"))
        elif args.export_type == "fp16":
            export_ir = engine.export(model=model, export_type=ExportType.OPENVINO, \
                                        export_root = args.output.joinpath(f"{class_name}/fp16"), \
                                        datamodule = datamodule, \
                                        compression_type=CompressionType.FP16)
        elif args.export_type == "int8":
            export_ir = engine.export(model=model, export_type=ExportType.OPENVINO, \
                                        export_root = args.output.joinpath(f"{class_name}/int8"), \
                                        datamodule = datamodule, \
                                        compression_type=CompressionType.INT8)
        elif args.export_type == "int8_ptq":
            export_ir = engine.export(model=model, export_type=ExportType.OPENVINO, \
                                        export_root = args.output.joinpath(f"{class_name}/int8_ptq"), \
                                        datamodule = datamodule, \
                                        compression_type=CompressionType.INT8_PTQ)
        elif args.export_type == "int8_acq":
            export_ir = engine.export(model=model, export_type=ExportType.OPENVINO, \
                                        export_root = args.output.joinpath(f"{class_name}/int8_acq"), \
                                        datamodule = datamodule, \
                                        compression_type=CompressionType.INT8_ACQ, metric='AUROC')
        else:
            raise ValueError(f"{args.export_type} is not supported export type")
        print(f"Export OpenVINO {args.export_type} IR: {export_ir}")