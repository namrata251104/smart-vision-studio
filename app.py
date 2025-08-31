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
        
        # Initialize MediaPipe for hands and pose (no face)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Advanced AI models
        self.yolo_model = None
        self.load_yolo_model()
        
        # Audio processing
        self.audio_stream = None
        self.audio_data = []
        self.is_recording_audio = False
        
        # Video recording
        self.video_writer = None
        self.is_recording_video = False
        self.recorded_frames = []
        
        # Advanced tracking
        self.object_tracker = cv2.TrackerCSRT_create()
        self.tracking_active = False
        self.tracked_bbox = None
        
        # Advanced features disabled for lightweight deployment
        # self.advanced_features = AdvancedFeatures()
        
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
            'pose_estimation': self.pose_estimation_mode,
            'environmental': self.environmental_mode
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
        """Initialize webcam capture - disabled for deployment"""
        print("Camera disabled for cloud deployment - using demo mode")
        return True
    
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
    
    def ai_detection_mode(self, frame):
        """AI detection disabled for lightweight deployment"""
        cv2.putText(frame, "AI Detection: Disabled for Cloud Deployment", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Use Object Detection mode instead", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
    
    def sound_visualization_mode(self, frame):
        """Sound visualization disabled for lightweight deployment"""
        cv2.putText(frame, "Sound Visualization: Disabled for Cloud", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame
    
    def video_recording_mode(self, frame):
        """Video recording with playback controls"""
        if self.is_recording_video:
            # Add recording indicator
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Store frame for recording
            self.recorded_frames.append(frame.copy())
            
            # Limit buffer size
            if len(self.recorded_frames) > 300:  # ~10 seconds at 30fps
                self.recorded_frames.pop(0)
        
        cv2.putText(frame, f"Recorded Frames: {len(self.recorded_frames)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def advanced_tracking_mode(self, frame):
        """Advanced object tracking with ML"""
        if not self.tracking_active:
            # Auto-detect objects to track
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.3, 10)
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            else:
                cv2.putText(frame, "Show your hand for gesture control", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show tracking info
        cv2.putText(frame, "Advanced Tracking: Feature Points", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def pose_estimation_mode(self, frame):
        """Full body pose estimation using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, "Pose Detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Stand in view for pose detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def sound_visualization_mode(self, frame):
        """Audio visualization with frequency bars"""
        # Create demo frequency bars
        bar_width = 20
        bar_gap = 5
        num_bars = 20
        
        for i in range(num_bars):
            # Simulate frequency data
            height = int(np.random.random() * 200 + 50)
            x = 50 + i * (bar_width + bar_gap)
            y = frame.shape[0] - height - 50
            
            # Color based on frequency
            color = (int(255 * i / num_bars), 255 - int(255 * i / num_bars), 128)
            cv2.rectangle(frame, (x, y), (x + bar_width, frame.shape[0] - 50), color, -1)
        
        cv2.putText(frame, "Sound Visualization (Demo)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
    
    def recognize_gesture(self, landmarks):
        """Simple gesture recognition based on hand landmarks"""
        if len(landmarks) < 21:
{{ ... }}
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                
                # Simple gesture recognition
                gesture = self.recognize_gesture(landmarks)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Execute gesture commands with cooldown
                if self.gesture_cooldown <= 0:
                    if "Stop Recording" in gesture and self.is_recording_video:
                        self.stop_video_recording()
                        self.gesture_cooldown = 30
                    elif "Start Recording" in gesture and not self.is_recording_video:
                        self.start_video_recording()
                        self.gesture_cooldown = 30
                
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        
        return frame
    
    def start_video_recording(self):
        """Start video recording"""
        self.is_recording_video = True
        self.recorded_frames = []
        print("Video recording started")
    
    def stop_video_recording(self):
        """Stop video recording and save"""
        if self.is_recording_video and len(self.recorded_frames) > 0:
            self.is_recording_video = False
            # Save video in background thread
            threading.Thread(target=self.save_recorded_video, daemon=True).start()
            print("Video recording stopped")
    
    def save_recorded_video(self):
        """Save recorded frames to video file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recorded_video_{timestamp}.mp4"
            
            if len(self.recorded_frames) > 0:
                height, width = self.recorded_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
                
                for frame in self.recorded_frames:
                    out.write(frame)
                
                out.release()
                print(f"Video saved as {filename}")
        except Exception as e:
            print(f"Error saving video: {e}")
    
    def initialize_audio(self):
        """Initialize audio capture for visualization"""
        try:
            self.audio_stream = pyaudio.PyAudio()
            return True
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            return False
    
    def update_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def generate_frames(self):
        """Generate video frames for streaming - demo mode for deployment"""
        import numpy as np
        
        while self.is_running:
            # Create a demo frame with gradient background
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create gradient background
            for i in range(480):
                for j in range(640):
                    frame[i, j] = [int(i/2), int(j/3), int((i+j)/4)]
            
            # Add demo text
            cv2.putText(frame, "Smart Vision Studio - Demo Mode", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Mode: {self.current_mode}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Filter: {self.current_filter}", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Camera disabled for cloud deployment", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "Connect your own camera for full functionality", (50, 430), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Apply current filter for demo
            if self.current_filter in self.filters and self.current_filter != 'none':
                try:
                    frame = self.filters[self.current_filter](frame)
                except:
                    pass  # Skip filter if it fails
            
            # Add UI overlay
            frame = self.add_ui_overlay(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Calculate FPS
            current_time = time.time()
            self.current_fps = 30  # Fixed FPS for demo
            self.last_frame_time = current_time
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(1/30)  # 30 FPS
    
    def add_ui_overlay(self, frame):
        """Add enhanced UI overlay with stats"""
        # Mode indicator
        cv2.putText(frame, f"Mode: {self.current_mode.replace('_', ' ').title()}", 
                   (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps}", 
                   (frame.shape[1] - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recording indicator
        if self.is_recording_video:
            cv2.circle(frame, (frame.shape[1] - 30, 90), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 55, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def pose_estimation_mode(self, frame):
        """3D pose estimation and body tracking"""
        return self.advanced_features.pose_estimation_mode(frame)
    
    def edge_computing_mode(self, frame):
        """Edge computing optimized processing"""
        return self.advanced_features.edge_computing_mode(frame)
    
    def ai_model_switching_mode(self, frame):
        """Dynamic AI model switching"""
        return self.advanced_features.ai_model_switching_mode(frame)
    
    def performance_analytics_mode(self, frame):
        """Real-time performance analytics"""
        return self.advanced_features.performance_analytics_mode(frame)
    
    def background_replacement_mode(self, frame):
        """AI background replacement"""
        return self.advanced_features.smart_background_replacement(frame)
    
    def object_analytics_mode(self, frame):
        """Advanced object counting and analytics"""
        return self.advanced_features.object_counting_analytics(frame)
    
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
