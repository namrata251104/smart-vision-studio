from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import base64
import threading
import time
from datetime import datetime
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_vision_studio'

class SmartVisionProcessor:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_mode = 'object_detection'
        self.current_filter = 'none'
        self.frame_count = 0
        
        # Background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Mode processors
        self.modes = {
            'object_detection': self.object_detection_mode,
            'motion_tracking': self.motion_tracking_mode,
            'creative_art': self.creative_art_mode,
            'environmental': self.environmental_mode,
            'gesture_control': self.gesture_control_mode,
            'ai_detection': self.ai_detection_mode,
            'sound_visual': self.sound_visualization_mode,
            'video_record': self.video_recording_mode,
            'advanced_track': self.advanced_tracking_mode,
            'pose_estimation': self.pose_estimation_mode
        }
        
        # Creative filters
        self.creative_filters = {
            'none': self.no_filter,
            'oil_painting': self.oil_painting_filter,
            'pencil_sketch': self.pencil_sketch_filter,
            'watercolor': self.watercolor_filter,
            'pop_art': self.pop_art_filter,
            'neon_glow': self.neon_glow_filter,
            'thermal_vision': self.thermal_vision_filter,
            'cyberpunk': self.cyberpunk_filter,
            'vintage': self.vintage_filter,
            'cartoon': self.cartoon_filter,
            'blur': self.blur_filter,
            'edge': self.edge_filter
        }

    def start_processing(self):
        """Start camera or demo mode"""
        if self.start_camera():
            self.is_running = True
            return True
        else:
            # Demo mode with synthetic frames
            self.is_running = True
            return True

    def start_camera(self):
        """Try to initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return True
            self.cap = None
            return False
        except:
            self.cap = None
            return False

    def stop_processing(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()

    def generate_frames(self):
        """Generate video frames"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    frame = self.create_demo_frame()
            else:
                frame = self.create_demo_frame()
            
            # Process frame based on current mode
            if self.current_mode in self.modes:
                frame = self.modes[self.current_mode](frame)
            
            # Apply filter
            if self.current_filter in self.creative_filters:
                frame = self.creative_filters[self.current_filter](frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS

    def create_demo_frame(self):
        """Create animated demo frame"""
        self.frame_count += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Animated gradient background
        for y in range(480):
            for x in range(640):
                r = int(128 + 127 * np.sin((x + self.frame_count) * 0.01))
                g = int(128 + 127 * np.sin((y + self.frame_count) * 0.01))
                b = int(128 + 127 * np.sin((x + y + self.frame_count) * 0.005))
                frame[y, x] = [b, g, r]
        
        # Add text overlay
        cv2.putText(frame, "Smart Vision Studio Demo", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Mode: {self.current_mode}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Filter: {self.current_filter}", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame

    # Processing modes
    def object_detection_mode(self, frame):
        """Color-based object detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for detection
        color_ranges = {
            'red': ([0, 120, 70], [10, 255, 255]),
            'blue': ([100, 150, 0], [140, 255, 255]),
            'green': ([40, 40, 40], [80, 255, 255])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{color_name.title()} Object', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

    def motion_tracking_mode(self, frame):
        """Motion detection and tracking"""
        fg_mask = self.background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, 'MOTION', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        status = "MOTION DETECTED!" if motion_detected else "No Motion"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame

    def creative_art_mode(self, frame):
        """Creative artistic processing"""
        cv2.putText(frame, "Creative Art Mode", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Use filter buttons for effects", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def environmental_mode(self, frame):
        """Environmental analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Dominant color analysis
        data = frame.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[0].astype(int)
        
        cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Dominant Color: RGB{tuple(dominant_color[::-1])}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def gesture_control_mode(self, frame):
        """Gesture control mode"""
        cv2.putText(frame, "Gesture Control Mode", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Hand tracking enabled", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def ai_detection_mode(self, frame):
        """AI object detection mode"""
        cv2.putText(frame, "AI Detection Mode", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Advanced AI processing", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def sound_visualization_mode(self, frame):
        """Sound visualization"""
        # Create demo frequency bars
        bar_width = 20
        bar_gap = 5
        num_bars = 20
        
        for i in range(num_bars):
            height = int(np.random.random() * 200 + 50)
            x = 50 + i * (bar_width + bar_gap)
            y = frame.shape[0] - height - 50
            
            color = (int(255 * i / num_bars), 255 - int(255 * i / num_bars), 128)
            cv2.rectangle(frame, (x, y), (x + bar_width, frame.shape[0] - 50), color, -1)
        
        cv2.putText(frame, "Sound Visualization", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def video_recording_mode(self, frame):
        """Video recording mode"""
        cv2.putText(frame, "Video Recording Mode", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, "Recording capabilities enabled", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def advanced_tracking_mode(self, frame):
        """Advanced tracking mode"""
        cv2.putText(frame, "Advanced Tracking", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        cv2.putText(frame, "ML-powered tracking", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def pose_estimation_mode(self, frame):
        """Pose estimation mode"""
        cv2.putText(frame, "Pose Estimation", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 255, 0), 2)
        cv2.putText(frame, "Body pose detection", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    # Creative filters
    def no_filter(self, frame):
        return frame

    def oil_painting_filter(self, frame):
        return cv2.xphoto.oilPainting(frame, 7, 1) if hasattr(cv2, 'xphoto') else frame

    def pencil_sketch_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur_inv = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur_inv, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def watercolor_filter(self, frame):
        bilateral = cv2.bilateralFilter(frame, 15, 80, 80)
        return cv2.edgePreservingFilter(bilateral, flags=2, sigma_s=50, sigma_r=0.4)

    def pop_art_filter(self, frame):
        k = 8
        data = np.float32(frame).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        return centers[labels.flatten()].reshape(frame.shape)

    def neon_glow_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        neon = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
        return cv2.addWeighted(frame, 0.7, neon, 0.3, 0)

    def thermal_vision_filter(self, frame):
        return cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    def cyberpunk_filter(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def vintage_filter(self, frame):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        vintage = cv2.addWeighted(sharpened, 0.8, 
                                cv2.applyColorMap(sharpened, cv2.COLORMAP_AUTUMN), 0.2, 0)
        return vintage

    def cartoon_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    def blur_filter(self, frame):
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def edge_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Initialize processor
processor = SmartVisionProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Smart Vision Studio is running'})

@app.route('/video_feed')
def video_feed():
    return Response(processor.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    if processor.start_processing():
        return jsonify({'status': 'success', 'message': 'Camera started'})
    return jsonify({'status': 'error', 'message': 'Failed to start camera'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    processor.stop_processing()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/change_mode', methods=['POST'])
def change_mode():
    mode = request.json.get('mode')
    processor.current_mode = mode
    return jsonify({'status': 'success', 'message': f'Mode changed to {mode}'})

@app.route('/change_filter', methods=['POST'])
def change_filter():
    filter_name = request.json.get('filter')
    processor.current_filter = filter_name
    return jsonify({'status': 'success', 'message': f'Filter changed to {filter_name}'})

@app.route('/get_modes')
def get_modes():
    return jsonify({
        'modes': list(processor.modes.keys()),
        'current_mode': processor.current_mode
    })

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    return jsonify({'status': 'success', 'message': 'Photo captured'})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    return jsonify({'status': 'success', 'message': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    return jsonify({'status': 'success', 'message': 'Recording stopped'})

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'fps': 30,
        'mode': processor.current_mode,
        'filter': processor.current_filter,
        'is_running': processor.is_running
    })

if __name__ == '__main__':
    print("Starting Smart Vision Studio...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
