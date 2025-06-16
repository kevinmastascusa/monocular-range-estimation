import cv2
import numpy as np
import torch
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms

# Initialize ONNX Runtime sessions
yolo_session = onnxruntime.InferenceSession("models/yolov5n.onnx")
midas_session = onnxruntime.InferenceSession("models/midas_small_v3.onnx")

# Load YOLOv5 model to get class names
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
yolo_classes = yolo_model.names

def preprocess_yolo(frame):
    # Resize to 640x640
    img = cv2.resize(frame, (640, 640))
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    img = img.astype(np.float32) / 255.0
    # Transpose to NCHW format
    img = img.transpose(2, 0, 1)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_midas(frame):
    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to MiDaS small v3 input size
    img = cv2.resize(img, (384, 384))
    
    # Convert to float32 and normalize
    img = img.astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Transpose to NCHW format
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension and ensure float32
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # Convert boxes to [x1, y1, x2, y2] format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while indices.size > 0:
        # Pick the highest score
        i = indices[0]
        keep.append(i)
        
        if indices.size == 1:
            break
            
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate IoU
        iou = intersection / (areas[i] + areas[indices[1:]] - intersection)
        
        # Keep boxes with IoU less than threshold
        indices = indices[1:][iou < iou_threshold]
    
    return keep

def process_frame(frame):
    # Get original frame dimensions
    h, w = frame.shape[:2]
    
    # Run YOLOv5 inference
    yolo_inputs = {yolo_session.get_inputs()[0].name: preprocess_yolo(frame)}
    yolo_outputs = yolo_session.run(None, yolo_inputs)
    
    # Process YOLOv5 outputs
    predictions = yolo_outputs[0][0]  # First batch, first image
    boxes = []
    scores = []
    class_ids = []
    
    # Process predictions with higher confidence threshold
    for pred in predictions:
        confidence = pred[4]
        if confidence > 0.45:  # Increased confidence threshold
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > 0.45:  # Increased class score threshold
                # Convert normalized coordinates to original image size
                # YOLO outputs are in [x_center, y_center, width, height] format
                x_center = pred[0] * w / 640
                y_center = pred[1] * h / 640
                width = pred[2] * w / 640
                height = pred[3] * h / 640
                
                # Convert to [x1, y1, x2, y2] format
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(class_score))
                class_ids.append(class_id)
    
    # Apply NMS with stricter IoU threshold
    if boxes:
        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = non_max_suppression(boxes, scores, iou_threshold=0.3)  # Stricter IoU threshold
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = [class_ids[i] for i in keep]
    
    # Run MiDaS inference
    midas_input = preprocess_midas(frame)
    midas_outputs = midas_session.run(None, {midas_session.get_inputs()[0].name: midas_input})
    
    # Get depth map and reshape if needed
    depth_map = midas_outputs[0]
    if len(depth_map.shape) == 4:  # If output is [1, 1, H, W]
        depth_map = depth_map[0, 0]
    elif len(depth_map.shape) == 3:  # If output is [1, H, W]
        depth_map = depth_map[0]
    
    # Debug prints for depth map
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth map min: {depth_map.min()}, max: {depth_map.max()}")
    print(f"Depth map mean: {depth_map.mean()}")
    
    # Process depth map
    depth_map = cv2.resize(depth_map, (w, h))
    
    # Normalize depth map
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    print(f"After resize - min: {depth_min}, max: {depth_max}")
    
    # Normalize to [0, 1] range (darker = closer)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    # Convert to visualization format
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    
    # Create a copy of the frame for drawing
    frame_with_boxes = frame.copy()
    
    # Draw bounding boxes and depth information
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        # Get average depth in the bounding box
        box_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(box_depth)
        
        # Draw bounding box
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with class name and depth
        label = f"{yolo_classes[class_id]}: {avg_depth:.1f}"
        cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame_with_boxes, depth_map

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_with_boxes, depth_map = process_frame(frame)
        
        # Show frames in separate windows
        cv2.imshow('Object Detection', frame_with_boxes)
        cv2.imshow('Depth Map', depth_map)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 