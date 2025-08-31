from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import base64
import threading
import time
from datetime import datetime
import json
import os
import mediapipe as mp
# Lightweight deployment - AI features disabled for cloud
# from ultralytics import YOLO
# import torch
# Lightweight deployment - minimal imports
import wave
import tempfile
# from advanced_features import AdvancedFeatures

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_vision_studio'

class SmartVisionProcessor:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_mode = 'object_detection'
        self.current_filter = 'none'
        
        # Initialize MediaPipe for gesture control
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Advanced AI models
        self.yolo_model = None
        self.load_yolo_model()
        
        # Simplified - no recording or advanced tracking
        
        # Advanced features disabled for lightweight deployment
        # self.advanced_features = AdvancedFeatures()
        
        # Simplified mode processors
        self.modes = {
            'object_detection': self.object_detection_mode,
            'motion_tracking': self.motion_tracking_mode,
            'creative_art': self.creative_art_mode,
            'environmental': self.environmental_mode,
            'gesture_control': self.gesture_control_mode
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
            'cartoon': self.cartoon_filter
        }
        
        # Motion tracking variables
        self.motion_threshold = 5000
        self.motion_history = []
        self.tracked_objects = {}
        
        # Environmental analysis
        self.color_zones = {}
        self.brightness_levels = []
        
        # Gesture recognition
        self.last_gesture = None
        self.gesture_cooldown = 0
        
        # Audio visualization
        self.audio_buffer = np.zeros(1024)
        self.freq_data = np.zeros(512)
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def load_yolo_model(self):
        """YOLO model disabled for lightweight deployment"""
        print("YOLO model disabled for cloud deployment")
        self.yolo_model = None
    
    def initialize_camera(self):
        """Initialize webcam capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print("Camera initialized successfully")
                return True
            else:
                print("Failed to open camera")
                return False
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def no_filter(self, frame):
        return frame
    
    def oil_painting_filter(self, frame):
        """Oil painting artistic effect"""
        return cv2.xphoto.oilPainting(frame, 7, 1) if hasattr(cv2, 'xphoto') else cv2.bilateralFilter(frame, 15, 80, 80)
    
    def pencil_sketch_filter(self, frame):
        """Pencil sketch effect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def watercolor_filter(self, frame):
        """Watercolor painting effect"""
        # Bilateral filter for smooth color regions
        smooth = cv2.bilateralFilter(frame, 15, 50, 50)
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Combine with reduced opacity
        return cv2.addWeighted(smooth, 0.8, edges, 0.2, 0)
    
    def pop_art_filter(self, frame):
        """Pop art effect with vibrant colors"""
        # Quantize colors
        data = frame.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        return segmented_data.reshape(frame.shape)
    
    def neon_glow_filter(self, frame):
        """Neon glow effect"""
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create glow effect
        glow = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        glow = cv2.GaussianBlur(glow, (15, 15), 0)
        
        # Apply neon colors
        neon = np.zeros_like(frame)
        neon[:, :, 0] = glow  # Blue channel
        neon[:, :, 1] = glow * 0.8  # Green channel
        neon[:, :, 2] = glow * 1.2  # Red channel
        
        return cv2.addWeighted(frame, 0.7, neon, 0.3, 0)
    
    def thermal_vision_filter(self, frame):
        """Thermal vision effect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    def cyberpunk_filter(self, frame):
        """Cyberpunk aesthetic with enhanced colors"""
        # Enhance specific color channels
        frame = cv2.addWeighted(frame, 1.2, frame, 0, 30)
        frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.5, 0, 255)  # Blue
        frame[:, :, 1] = np.clip(frame[:, :, 1] * 0.8, 0, 255)  # Green
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.3, 0, 255)  # Red
        return frame
    
    def vintage_filter(self, frame):
        """Vintage film effect"""
        # Add sepia tone
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        
        # Add noise
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        return cv2.add(sepia, noise)
    
    def cartoon_filter(self, frame):
        """Cartoon-like effect"""
        # Bilateral filter for smooth regions
        smooth = cv2.bilateralFilter(frame, 15, 50, 50)
        
        # Edge detection
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine
        return cv2.bitwise_and(smooth, edges)
    
    def detect_objects_simple(self, frame):
        """Simple object detection using contours and color"""
        objects_detected = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common objects
        color_ranges = {
            'red_object': ([0, 50, 50], [10, 255, 255]),
            'green_object': ([40, 50, 50], [80, 255, 255]),
            'blue_object': ([100, 50, 50], [130, 255, 255]),
            'yellow_object': ([20, 50, 50], [40, 255, 255])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    objects_detected.append({
                        'name': color_name,
                        'bbox': (x, y, w, h),
                        'confidence': min(area / 10000, 1.0)
                    })
        
        return objects_detected
    
    def object_detection_mode(self, frame):
        """Object detection and labeling"""
        objects = self.detect_objects_simple(frame)
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            confidence = obj['confidence']
            name = obj['name']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add object count
        cv2.putText(frame, f"Objects detected: {len(objects)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def motion_tracking_mode(self, frame):
        """Motion detection and tracking"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw motion rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "MOTION", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add motion status
        status = "MOTION DETECTED" if motion_detected else "NO MOTION"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show motion mask in corner
        mask_resized = cv2.resize(fg_mask, (160, 120))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = mask_colored
        
        return frame
    
    def creative_art_mode(self, frame):
        """Apply creative artistic filters"""
        if self.current_filter in self.creative_filters:
            frame = self.creative_filters[self.current_filter](frame)
        
        # Add filter name
        cv2.putText(frame, f"Filter: {self.current_filter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def environmental_mode(self, frame):
        """Environmental analysis - colors, brightness, etc."""
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Calculate dominant colors
        data = frame.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        
        # Display environmental info
        cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show dominant colors
        for i, color in enumerate(centers):
            cv2.rectangle(frame, (10, 50 + i * 30), (60, 75 + i * 30), 
                         (int(color[0]), int(color[1]), int(color[2])), -1)
            cv2.putText(frame, f"Color {i+1}", (70, 70 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def gesture_control_mode(self, frame):
        """Hand gesture recognition and control"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                
                # Recognize gesture
                gesture = self.recognize_gesture(landmarks)
                
                # Display gesture
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Execute gesture commands
                self.execute_gesture_command(gesture)
        else:
            cv2.putText(frame, "Show your hand for gesture control", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def recognize_gesture(self, landmarks):
        """Simple gesture recognition based on hand landmarks"""
        if len(landmarks) < 21:
            return "Unknown"
        
        # Simple finger counting based on landmark positions
        fingers_up = []
        
        # Thumb
        if landmarks[4][0] > landmarks[3][0]:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Other fingers
        for i in [8, 12, 16, 20]:
            if landmarks[i][1] < landmarks[i-2][1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        total_fingers = sum(fingers_up)
        
        if total_fingers == 0:
            return "Fist - Reset Mode"
        elif total_fingers == 1:
            return "Point - Object Detection"
        elif total_fingers == 2:
            return "Peace - Motion Tracking"
        elif total_fingers == 3:
            return "Three - Creative Mode"
        elif total_fingers == 4:
            return "Four - Environmental"
        elif total_fingers == 5:
            return "Open Hand - Capture Photo"
        else:
            return f"{total_fingers} Fingers"
    
    def execute_gesture_command(self, gesture):
        """Execute commands based on recognized gestures"""
        if "Object Detection" in gesture:
            self.current_mode = 'object_detection'
        elif "Motion Tracking" in gesture:
            self.current_mode = 'motion_tracking'
        elif "Creative Mode" in gesture:
            self.current_mode = 'creative_art'
        elif "Environmental" in gesture:
            self.current_mode = 'environmental'
        elif "Reset Mode" in gesture:
            self.current_mode = 'object_detection'
        elif "Capture Photo" in gesture:
            # Trigger photo capture
            print("Photo captured via gesture!")
    
    def update_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame(self):
        """Process a single frame from camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Apply current mode processing
        if self.current_mode in self.modes:
            frame = self.modes[self.current_mode](frame)
        
        # Add UI overlay
        frame = self.add_ui_overlay(frame)
        
        return frame
    
    def generate_frames(self):
        """Generate video frames for streaming"""
        while self.is_running:
            frame = self.process_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def add_ui_overlay(self, frame):
        """Add enhanced UI overlay with stats"""
        # Mode indicator
        cv2.putText(frame, f"Mode: {self.current_mode.replace('_', ' ').title()}", 
                   (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps}", 
                   (frame.shape[1] - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Simplified UI - no recording indicator
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    
    
    def generate_frames(self):
        """Generator function for video streaming"""
        while self.is_running:
            frame = self.process_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def start_processing(self):
        """Start the video processing"""
        if self.initialize_camera():
            self.is_running = True
            return True
        return False
    
    def stop_processing(self):
        """Stop the video processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()

# Global processor instance
processor = SmartVisionProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(processor.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    try:
        if processor.start_processing():
            print("Camera started successfully via API")
            return jsonify({'status': 'success', 'message': 'Camera started'})
        else:
            print("Camera failed to start - no camera detected")
            return jsonify({'status': 'error', 'message': 'No camera detected. Please check if camera is connected and not in use by another application.'})
    except Exception as e:
        print(f"Camera start error: {e}")
        return jsonify({'status': 'error', 'message': f'Camera error: {str(e)}'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    processor.stop_processing()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/change_mode', methods=['POST'])
def change_mode():
    mode = request.json.get('mode')
    if mode in processor.modes:
        processor.current_mode = mode
        return jsonify({'status': 'success', 'message': f'Mode changed to {mode}'})
    return jsonify({'status': 'error', 'message': 'Invalid mode'})

@app.route('/change_filter', methods=['POST'])
def change_filter():
    filter_name = request.json.get('filter')
    if filter_name in processor.creative_filters:
        processor.current_filter = filter_name
        return jsonify({'status': 'success', 'message': f'Filter changed to {filter_name}'})
    return jsonify({'status': 'error', 'message': 'Invalid filter'})

@app.route('/get_modes', methods=['GET'])
def get_modes():
    return jsonify({
        'modes': list(processor.modes.keys()),
        'filters': list(processor.creative_filters.keys()),
        'current_mode': processor.current_mode,
        'current_filter': processor.current_filter
    })

@app.route('/start_recording', methods=['POST'])
def start_recording():
    processor.start_video_recording()
    return jsonify({'status': 'success', 'message': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    processor.stop_video_recording()
    return jsonify({'status': 'success', 'message': 'Recording stopped'})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    return jsonify({
        'fps': processor.current_fps,
        'is_recording': processor.is_recording_video,
        'recorded_frames': len(processor.recorded_frames),
        'mode': processor.current_mode,
        'filter': processor.current_filter
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
