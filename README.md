# 🎥 Smart Vision Studio

**Real-time AI-powered image processing web application** with advanced computer vision features, artistic filters, and interactive controls.

## ✨ Features

### 🤖 AI Processing Modes
- **Object Detection** - Color-based detection with bounding boxes
- **AI Detection** - Advanced YOLOv8 object recognition
- **Motion Tracking** - Background subtraction and alerts
- **Pose Estimation** - 3D body tracking with MediaPipe
- **Gesture Control** - Hand gesture recognition
- **Environmental Analysis** - Brightness and color monitoring

### 🎨 Artistic Filters
- Oil Painting • Pencil Sketch • Watercolor • Pop Art
- Neon Glow • Thermal Vision • Cyberpunk • Vintage • Cartoon

### 🚀 Advanced Features
- Real-time video recording
- Sound visualization with FFT
- Performance analytics
- Background replacement
- Interactive web interface

## 🛠️ Tech Stack

- **Backend**: Flask, OpenCV, MediaPipe
- **AI/ML**: YOLOv8, PyTorch, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript
- **Audio**: PyAudio, SciPy

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser to http://localhost:5000
```

## 🎮 Controls

- **1-9**: Switch modes
- **R/T**: Start/Stop recording
- **Space**: Capture photo
- **P**: Pose estimation
- **B**: Background replacement

## 🌐 Deployment

Ready for cloud deployment with Railway, Render, or Heroku. Demo mode enabled for environments without camera access.

## 📄 License

MIT License

- Python 3.8+
- Webcam
- Windows/Mac/Linux
- Modern web browser

## 🔧 Customization

The application is modular and easily extensible:
- Add new filters in the `creative_filters` dictionary
- Implement new modes in the `modes` dictionary
- Customize object detection parameters
- Modify UI themes and layouts

## 🎯 Use Cases

- **Creative Content**: Generate artistic videos and photos
- **Security Monitoring**: Motion detection and alerts
- **Interactive Presentations**: Gesture-controlled demos
- **Environmental Analysis**: Monitor lighting and colors
- **Educational**: Learn computer vision concepts

Enjoy exploring the world of real-time computer vision! 🚀
