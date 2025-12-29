from ultralytics import YOLO

# Load the exported NCNN model
model = YOLO("yolo11n_ncnn_model/")

# Run inference on the webcam
# 'source=0' specifies the default local camera
# 'stream=True' returns a generator for memory efficiency
results = model.predict(source="0", show=True, imgsz=320, stream=True)

# Because stream=True, we must iterate through the results to actually run the loop
for r in results:
    pass  # The 'show=True' parameter handles the window display automatically