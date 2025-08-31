from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time

app = Flask(__name__)

class SimpleVisionProcessor:
    def __init__(self):
        self.is_running = False
        self.current_mode = 'demo'
        self.current_filter = 'none'
        
    def start_processing(self):
        self.is_running = True
        return True
        
    def stop_processing(self):
        self.is_running = False
        
    def generate_frames(self):
        """Generate demo frames for cloud deployment"""
        while self.is_running:
            # Create colorful demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Animated gradient
            t = time.time()
            for i in range(480):
                for j in range(640):
                    r = int(128 + 127 * np.sin(t + i * 0.01))
                    g = int(128 + 127 * np.sin(t + j * 0.01 + 2))
                    b = int(128 + 127 * np.sin(t + (i+j) * 0.005 + 4))
                    frame[i, j] = [b, g, r]
            
            # Add text overlay
            cv2.putText(frame, "Smart Vision Studio - Cloud Demo", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Mode: {self.current_mode}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Filter: {self.current_filter}", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Fully functional with local camera!", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Apply simple filter
            if self.current_filter == 'blur':
                frame = cv2.GaussianBlur(frame, (15, 15), 0)
            elif self.current_filter == 'edge':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(1/30)  # 30 FPS

processor = SimpleVisionProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(processor.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    if processor.start_processing():
        return jsonify({'status': 'success', 'message': 'Demo started'})
    return jsonify({'status': 'error', 'message': 'Failed to start demo'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    processor.stop_processing()
    return jsonify({'status': 'success', 'message': 'Demo stopped'})

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

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'fps': 30,
        'mode': processor.current_mode,
        'filter': processor.current_filter,
        'is_running': processor.is_running
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
