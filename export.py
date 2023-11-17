from ultralytics import YOLO

# Load a model
model = YOLO(r'yolov8s-seg.pt')  # load a custom trained

# Export the model
model.export(format='tensorrt')
