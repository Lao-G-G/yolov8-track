# yolov8-track

## 训练

如果只需要使用yolov8预训练的权重可以跳过。如果有自己的数据集，且与coco的类别相似可以冻结backbone只微调neck和head；如果与coco差别较大就得从头训练。

运行`train.py`

## 跟踪

跟踪是只在推理时进行帧关联的，不需要训练。

如果要使用BoT-SORT则运行`track_BoTSORT.py`，要使用ByteTrack则运行`track_ByteTrack.py`。
