from ultralytics import YOLO

#yolov8网络结构
#                   from  n    params  module                                       arguments                     
#  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
#  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
#  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
#  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
#  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
#  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
#  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
#  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
#  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
#  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]             #backbone                 
# 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
# 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
# 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
# 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
# 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
# 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
# 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
# 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
# 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
# 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
# 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
# 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]            #neck            
# 22        [15, 18, 21]  1    897664  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]     #head



def freeze_layer(trainer):
#冻结前num_freeze层
#    model = trainer.model
#    num_freeze = 10
#    print(f"Freezing {num_freeze} layers")
#    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
#    for k, v in model.named_parameters(): 
#        v.requires_grad = True  # train all layers 
#        if any(x in k for x in freeze): 
#            print(f'freezing {k}') 
#            v.requires_grad = False 
#    print(f"{num_freeze} layers are freezed.")
#冻结若干层
    model = trainer.model
    freeze_layers = [0,1,2,3,4,5,6,7,8,9]
    print(f"Freezing {freeze_layers} layers")
    freeze = [f'model.{x}.' for x in freeze_layers]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{freeze_layers} layers are freezed.")



# Create a new YOLO model from scratch (从头训练)
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)(使用预训练模型)
model = YOLO('yolov8n.pt')

model.add_callback("on_train_start", freeze_layer)

#print(model)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#results = model.train(data='self.yaml', epochs=3, device=[0,1])
results = model.train(data='self.yaml',
                      epochs=100,
                      device=0,
                      project='runs/detect',
                      name='frozen_backbone',)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
#success = model.export(format='onnx')

