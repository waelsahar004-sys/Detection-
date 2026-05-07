# YOLO Webcam Inference Project

A production-ready real-time object detection system using YOLOv8 and webcam streams. This project follows best practices for Python development including virtual environments, modular configuration, and comprehensive logging.

## 🎯 Features

- ✨ **Real-time Detection**: Process webcam stream at 30+ FPS
- 🎯 **YOLOv8 Support**: All model variants (nano to extra-large)
- 💻 **CPU & GPU**: Automatic device detection and support
- 🔧 **Highly Configurable**: Centralized config file for easy customization
- 📊 **Snapshot Capture**: Save frames with detections
- ⏸️ **Interactive Controls**: Pause, resume, and save functionality
- 📝 **Comprehensive Logging**: File and console logging
- 🐍 **Virtual Environment**: Isolated dependencies (best practice)

## 📋 Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows
- **Hardware**: Webcam/camera device

## 🚀 Quick Start

### 1. Automated Setup (Recommended)

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
setup.bat
```

### 2. Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Download model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Run Inference

```bash
# Basic usage (CPU, nano model)
python webcam_inference.py

# With GPU
python webcam_inference.py --device cuda

# Custom confidence threshold
python webcam_inference.py --conf 0.6

# Larger model
python webcam_inference.py --model yolov8m.pt

# Multiple options
python webcam_inference.py --model yolov8s.pt --device cuda --conf 0.5
```

## 🎮 Controls During Inference

| Key | Action |
|-----|--------|
| `q` | Quit inference |
| `s` | Save current frame with detections |
| `p` | Pause/Resume inference |

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "yolov8n.pt"  # nano (fastest, least accurate)
DEVICE = "cpu"              # cpu or cuda
CONFIDENCE_THRESHOLD = 0.5  # 0.0-1.0

# Webcam
CAMERA_INDEX = 0            # 0 for default camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Display
SHOW_BOXES = True
SHOW_LABELS = True
SHOW_FPS = True
LINE_THICKNESS = 2

# Output
OUTPUT_DIR = "detections"
SAVE_DETECTIONS = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "inference.log"
```

## 📊 YOLOv8 Model Variants

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **yolov8n.pt** | 3.2M | ⚡⚡⚡ Fastest | ⭐⭐⭐ | Real-time, limited hardware |
| **yolov8s.pt** | 11M | ⚡⚡ Fast | ⭐⭐⭐⭐ | Balanced |
| **yolov8m.pt** | 26M | ⚡ Medium | ⭐⭐⭐⭐⭐ | Production |
| **yolov8l.pt** | 52M | 🐢 Slow | ⭐⭐⭐⭐⭐ | High accuracy |
| **yolov8x.pt** | 135M | 🐢🐢 Very Slow | 🌟 Best | Maximum accuracy |

## 📁 Project Structure

```
.
├── webcam_inference.py    # Main inference script
├── config.py              # Configuration file
├── requirements.txt       # Python dependencies
├── setup.sh              # Linux/macOS setup script
├── setup.bat             # Windows setup script
├── README.md             # This file
├── .gitignore            # Git ignore patterns
├── venv/                 # Virtual environment (created after setup)
├── detections/           # Saved frames (created when saving)
└── inference.log         # Logging output
```

## 🔍 Troubleshooting

### Webcam Not Found
```bash
# Linux: List available cameras
ls -l /dev/video*

# Use specific camera (e.g., camera 1)
python webcam_inference.py --camera 1
```

### CUDA Not Available
```bash
# Make sure NVIDIA drivers are installed
nvidia-smi

# If errors persist, use CPU
python webcam_inference.py --device cpu
```

### Memory Issues
```bash
# Use smaller model
python webcam_inference.py --model yolov8n.pt --device cpu

# Reduce frame resolution in config.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

### Slow Performance
1. Use smaller model (`yolov8n.pt`)
2. Reduce frame resolution
3. Use GPU if available
4. Lower FPS setting in config

## 📝 Logging

Logs are saved to `inference.log` and printed to console:

```
2026-05-07 10:30:45,123 - INFO - Loading model: yolov8n.pt on cpu
2026-05-07 10:30:48,456 - INFO - Model loaded successfully
2026-05-07 10:30:48,789 - INFO - Webcam opened - Resolution: 1280x720
```

Change log level in `config.py`:
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages

## 🤝 Best Practices Used

1. **Virtual Environment**: Isolated dependencies
2. **Modular Configuration**: Centralized settings
3. **Structured Logging**: File and console logging
4. **Error Handling**: Graceful error recovery
5. **Code Organization**: Clean, well-documented code
6. **Type Hints**: Clear function signatures
7. **Resource Management**: Proper cleanup

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🎓 Next Steps

- Customize for specific object classes
- Integrate with video recording
- Add multi-camera support
- Implement detection alerts/notifications
- Deploy on edge devices (Raspberry Pi, NVIDIA Jetson)

---

**Created**: 2026-05-07 | **Status**: Production Ready ✅
