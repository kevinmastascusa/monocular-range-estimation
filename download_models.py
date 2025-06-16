import torch
import os

def download_models():
    print("Downloading YOLOv5n model...")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    dummy_input = torch.randn(1, 3, 384, 384)
    torch.onnx.export(yolo_model, (dummy_input,), 'models/yolov5n.onnx')
    print("YOLOv5n model downloaded and converted to ONNX")

    print("\nDownloading MiDaS model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    torch.onnx.export(midas, (dummy_input,), 'models/midas_small.onnx')
    print("MiDaS model downloaded and converted to ONNX")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    download_models() 