from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export to NCNN format
model.export(format="ncnn", half=True, imgsz=320, simplify=True)  # creates '/yolo11n_ncnn_model'