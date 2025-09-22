# Smart Security System

A real-time security monitoring system using computer vision and deep learning for object detection, tracking, and anomaly detection.

## Features

- **Real-time Object Detection** - YOLO v8 for person and vehicle detection
- **Multi-Object Tracking** - Track objects with unique IDs and trails
- **Motion Detection** - Background subtraction for motion analysis
- **Anomaly Detection** - Alerts for crowds and suspicious activity
- **Live Analytics** - FPS, detection counts, and performance metrics

## Configuration

Basic settings in the code:
- `confidence_threshold = 0.6` - Detection confidence
- `alert_threshold = 4` - Number of people for crowd alert
- `video_source = 0` - Use webcam (0) or video file path

## Key Components

- **Object Detection**: YOLO v8 with OpenCV integration
- **Tracking**: DeepSORT algorithm with trajectory visualization
- **Motion Analysis**: MOG2 background subtractor
- **Performance**: 30+ FPS on GPU, real-time processing

## Output

### Demo Video : [See the Demo](https://drive.google.com/file/d/1X2sAhrO9PLuXM3M281Lib6x3I7Zk5ITP/view?usp=sharing)

The system displays:
- Live video with detection boxes and tracking trails
- Real-time statistics overlay (FPS, object counts, motion %)
- Alert notifications for anomalies
- JSON log files with detection data

## Performance

- Detection Accuracy: 90%+ for persons and vehicles
- Processing Speed: 30+ FPS (GPU), 8+ FPS (CPU)
- Memory Usage: ~2GB GPU VRAM


## Technical Challenges & Solutions

### Challenge 1: Real-time Performance Optimization
**Problem**: Initial implementation was processing at only 5-8 FPS due to inefficient frame handling.
**Solution**: Implemented frame buffering and optimized the detection pipeline by reducing unnecessary computations and using proper OpenCV threading.

### Challenge 2: False Positive Reduction in Motion Detection
**Problem**: Background subtraction was generating too many false alerts from lighting changes and shadows.
**Solution**: Fine-tuned MOG2 parameters and added morphological operations to filter noise. Implemented temporal consistency checks to validate detections across multiple frames.

## Requirements

- Python 3.8+
- CUDA GPU (recommended)
- Webcam or video files

---

Built with YOLO v8, OpenCV, and PyTorch for enterprise-grade security monitoring.
