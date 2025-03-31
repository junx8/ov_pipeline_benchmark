# ov_pipeline_benchmark

## Test for the anomalib-patchcore-resnet18 model with mvtecad-bottle

### Test with CPU

```
python throughput_benchmark.py
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


### Test with GPU

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