# 🎨 Smart Vision Studio

A real-time multi-modal image processing application that combines object detection, motion tracking, creative artistic filters, environmental analysis, and gesture control - all without face processing.

## ✨ Features

### 🎯 **Object Detection Mode**
- Real-time color-based object detection
- Bounding box visualization with confidence scores
- Object counting and classification
- Support for multiple objects simultaneously

### 🏃 **Motion Tracking Mode**
- Advanced background subtraction
- Motion area highlighting
- Security monitoring capabilities
- Motion history tracking

### 🎨 **Creative Art Mode**
- Oil painting effect
- Pencil sketch transformation
- Watercolor simulation
- Pop art style
- Neon glow effects
- Thermal vision
- Cyberpunk aesthetic
- Vintage film look
- Cartoon stylization

### 🌍 **Environmental Analysis Mode**
- Real-time brightness monitoring
- Dominant color extraction
- Environmental condition assessment
- Color palette visualization

### 👋 **Gesture Control Mode**
- Hand gesture recognition using MediaPipe
- Interactive controls via hand movements
- Real-time gesture feedback
- Multiple gesture commands

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Open Browser**
   - Navigate to `http://localhost:5000`
   - Click "Start Camera" to begin
   - Switch between modes and filters in real-time

## 🎮 Controls

### Web Interface
- **Start/Stop Camera**: Control webcam access
- **Mode Buttons**: Switch between processing modes
- **Filter Buttons**: Apply creative effects
- **Capture Photo**: Save current frame

### Keyboard Shortcuts
- `1-5`: Switch between modes
- `Space`: Capture photo
- `S`: Start/Stop camera

## 🛠️ Technical Stack

- **Backend**: Flask + OpenCV + MediaPipe
- **Frontend**: Modern HTML5 + CSS3 + JavaScript
- **Computer Vision**: OpenCV for image processing
- **AI**: MediaPipe for gesture recognition
- **Real-time**: WebSocket streaming

## 📋 System Requirements

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
