# yolov8-track

## 训练

如果只需要使用yolov8预训练的权重可以跳过。

如果有自己的数据集，且与coco的类别相似可以冻结backbone只微调neck和head

```
model = YOLO('yolov8n.pt')
model.add_callback("on_train_start", freeze_layer)
```

如果与coco差别较大就得从头训练。

```
model = YOLO('yolov8n.yaml')
```

运行`train.py`

## 跟踪

跟踪是只在推理时进行帧关联的，不需要训练。

如果要使用BoT-SORT则运行`track_BoTSORT.py`,要使用ByteTrack则运行`track_ByteTrack.py`.

## 结果

3080上分辨率384x640平均每帧用时:

| **Method** | **Preprocess** | **Inference** | **Postprocess** |
| -----------| -------------- | ------------- | --------------- |
| BoTSORT    | 1ms            | 6.0ms         | 1.0ms           |
| ByteTrack  | 1ms            | 6.2ms         | 1.4ms           |

