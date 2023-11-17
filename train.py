from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

# # Train the model
# model.train(data='pts5-pose.yaml', epochs=300, imgsz=640)

# # Load a model
model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data='/home/evolution/Code/PTS5-Y8-Alpha/datasets', epochs=300, imgsz=640,device="0",workers=2)
results = model.train(data='/home/evolution/Code/PTS5-Y8-Alpha/datasets', epochs=300, imgsz=640,workers=8,batch = 72)
