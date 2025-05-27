# To use cv2 with conda, ensure you have installed OpenCV via:
# conda install -c conda-forge opencv
import cv2
import torch  # If using conda, install with: conda install pytorch torchvision torchaudio -c pytorch
import numpy as np # Add this line

# Load pre-trained YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True) # Old line
model = torch.hub.load('/Users/kevinmastascusa/yolov5', 'yolov5s', source='local') # Use local source, force_reload=False (default)

# Known object width in cm (example: stop sign ~70 cm)
KNOWN_WIDTH_CM = 70
FOCAL_LENGTH = 800  # needs calibration (pixels)

def estimate_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure the frame is a C-contiguous array
    frame_contiguous = np.ascontiguousarray(frame)
    results = model(frame_contiguous)
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        width_px = x2 - x1
        
        # Estimate distance if the target object is detected (e.g., 'stop sign')
        if label == 'person':  # change as needed
            distance_cm = estimate_distance(KNOWN_WIDTH_CM, FOCAL_LENGTH, width_px)
            cv2.putText(frame, f"{label} {distance_cm:.1f}cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Distance-Aware Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()