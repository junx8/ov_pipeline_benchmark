# ov_pipeline_benchmark

## Env Setup

### Install the Dependency

```
conda create -n ovplb python=3.11
conda activate ovplb
pip install anomalib
anomalib install --option full
```

### Prepare the dataset

- Download the dataset from [MVTecAD](https://www.mvtec.com/company/research/datasets) and put the data as follows:

```
datasets
└── MVTecAD
    ├── bottle
    ├── cable
    ├── capsule
    ├── carpet
    ├── grid
    ├── hazelnut
    ├── leather
    ├── license.txt
    ├── metal_nut
    ├── pill
    ├── readme.txt
    ├── screw
    ├── tile
    ├── toothbrush
    ├── transistor
    ├── wood
    └── zipper
```

<details>
<summary><b>Export OpenVINO IR (Optional)</b></summary>
    
### Prepare the Model(Optional)

```
python export_ir.py -et -cn [your_category]
```

> It will download the weights from huggingface, please make sure you can access it.

```
ovmodels/
├── patchcore_resnet18_bottle
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_cable
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_capsule
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_carpet
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_grid
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_hazelnut
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_leather
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_metal_nut
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_pill
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_screw
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_tile
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_toothbrush
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_transistor
│   ├── model.bin
│   └── model.xml
├── patchcore_resnet18_wood
│   ├── model.bin
│   └── model.xml
└── patchcore_resnet18_zipper
    ├── model.bin
    └── model.xml
```
</details>

## Run Benchmark

### Run Benchmark on CPU

```
python throughput_benchmark.py
``` 
or



```
python throughput_benchmark.py -cn bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
```

```
INFO:root:OpenVINO:
INFO:root:Build ................................. 2025.0.0-17942-1f68be9f594-releases/2025/0

Starting Performance Testing...

Performance:
 - 83 images task 2.43 sec, 0.029 sec/image
 - throughput 34.15 FPS

Saving the results...

Saving Results Performance:
 - 83 images task 8.36 sec, 0.101 sec/image
 - throughput 9.93 FPS
```


### Run Benchmark on GPU

```
python throughput_benchmark.py -d GPU
```

```
INFO:root:OpenVINO:
INFO:root:Build ................................. 2025.0.0-17942-1f68be9f594-releases/2025/0

Starting Performance Testing...

Performance:
 - 83 images task 1.98 sec, 0.024 sec/image
 - throughput 41.95 FPS

Saving the results...

Saving Results Performance:
 - 83 images task 5.00 sec, 0.060 sec/image
 - throughput 16.59 FPS
```


## Run Demo

```
python demo.py
```

```
python demo.py -d GPU
```

> The `ovmodels/` only provides the OpenVINO IR for bottle. IRs of other categories can be downloaded [here]().
