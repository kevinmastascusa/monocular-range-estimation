# 🎯 Monocular Range Estimation

A sophisticated real-time object detection and depth estimation system that combines state-of-the-art YOLOv5 for object detection with MiDaS for depth estimation. This project enables accurate object detection and relative distance estimation using a single camera, making it ideal for applications in robotics, autonomous systems, and computer vision research.

> **Author:** Kevin Mastascusa  
> **Last Updated:** June 15, 2024 9:54 PM

## 📜 Project Evolution

The project began as a geometric-based distance estimation system using focal length calculations, where:
- Distance was calculated using the relationship between object size, focal length, and image plane
- Basic computer vision techniques were used for object detection
- Simple geometric approximations were employed for depth estimation

It has since evolved into a more sophisticated system that:
- Uses deep learning for both object detection and depth estimation
- Provides more accurate and robust distance measurements
- Handles complex scenes and varying lighting conditions
- Offers real-time performance with modern neural networks

## ✨ Features

- 🎥 Real-time object detection using YOLOv5
- 📏 High-precision depth estimation using MiDaS
- 🎨 Combined visualization of object detection and depth map
- 📊 Distance estimation for detected objects
- 📸 Support for webcam input
- 🚀 Optimized for real-time performance

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- PyTorch
- ONNX Runtime
- Other dependencies listed in `requirements.txt`

## 📥 Installation

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

## 🚀 Usage

Run the main script to start object detection and depth estimation:
```bash
python ocular_core/distance_aware_object_detection.py
```

### 🎮 Controls
- Press 'q' to quit the application
- The application will show two windows:
  - Object Detection: Shows the video feed with bounding boxes and distance estimates
  - Depth Map: Shows the depth estimation visualization

## 📁 Project Structure

```
monocular-range-estimation/
├── ocular_core/
│   └── distance_aware_object_detection.py  # Main application code
├── models/                                 # Downloaded model files
├── requirements.txt                        # Python dependencies
├── download_models.py                      # Model download script
└── README.md                              # This file
```

## 🔍 How It Works

1. **Object Detection** 🔍
   - Uses YOLOv5 to detect objects in the video stream
   - Processes frames in real-time
   - Applies confidence thresholds for accurate detection

2. **Depth Estimation** 📏
   - Uses MiDaS to estimate depth for each frame
   - Generates high-quality depth maps
   - Maintains temporal consistency

3. **Distance Calculation** 📊
   - Combines detection and depth information
   - Estimates relative distances to objects
   - Provides real-time distance updates

4. **Visualization** 🎨
   - Displays bounding boxes with distance estimates
   - Shows depth map with color-coded distances
   - Updates in real-time

## 🤖 Model Information

### YOLOv5
- **Model**: YOLOv5n (nano version)
- **Input Size**: 640x640
- **Confidence Threshold**: 0.45
- **Performance**: Optimized for real-time inference

### MiDaS
- **Model**: MiDaS small v3
- **Input Size**: 384x384
- **Output**: Relative depth map
- **Features**: High-quality depth estimation

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Submit issues and enhancement requests
- Fork the repository and create pull requests
- Improve documentation
- Add new features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv5 team for the excellent object detection model
- MiDaS team for the depth estimation model
- OpenCV community for the computer vision tools
