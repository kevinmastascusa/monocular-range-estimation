# Monocular Range Estimation

A real-time object detection and depth estimation system that combines YOLOv5 for object detection with MiDaS for depth estimation. This project allows you to detect objects in a video stream and estimate their relative distances using a single camera.

## Features

- Real-time object detection using YOLOv5
- Depth estimation using MiDaS
- Combined visualization of object detection and depth map
- Distance estimation for detected objects
- Support for webcam input

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- ONNX Runtime
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kevinmastascusa/monocular-range-estimation.git
cd monocular-range-estimation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python download_models.py
```

## Usage

Run the main script to start object detection and depth estimation:
```bash
python ocular_core/distance_aware_object_detection.py
```

### Controls
- Press 'q' to quit the application
- The application will show two windows:
  - Object Detection: Shows the video feed with bounding boxes and distance estimates
  - Depth Map: Shows the depth estimation visualization

## Project Structure

```
monocular-range-estimation/
├── ocular_core/
│   └── distance_aware_object_detection.py  # Main application code
├── models/                                 # Downloaded model files
├── requirements.txt                        # Python dependencies
├── download_models.py                      # Model download script
└── README.md                              # This file
```

## How It Works

1. **Object Detection**: Uses YOLOv5 to detect objects in the video stream
2. **Depth Estimation**: Uses MiDaS to estimate depth for each frame
3. **Distance Calculation**: Combines detection and depth information to estimate object distances
4. **Visualization**: Displays bounding boxes with distance estimates and a depth map

## Model Information

- **YOLOv5**: Used for object detection
  - Model: YOLOv5n (nano version)
  - Input size: 640x640
  - Confidence threshold: 0.45

- **MiDaS**: Used for depth estimation
  - Model: MiDaS small v3
  - Input size: 384x384
  - Output: Relative depth map

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
