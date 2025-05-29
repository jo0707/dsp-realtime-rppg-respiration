import sys
import cv2
import numpy as np
import time
import mediapipe as mp
from collections import deque
from scipy import signal
from scipy.signal import savgol_filter
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# Global variables
fps = 30
buffer_size = 300  # 10 seconds
min_window = int(1.6 * fps)  # 1.6 seconds for POS

# Data buffers
rgb_buffer = deque(maxlen=buffer_size)
pos_raw_buffer = deque(maxlen=buffer_size)
pos_filtered_buffer = deque(maxlen=buffer_size)
pos_savgol_buffer = deque(maxlen=buffer_size)
shoulder_buffer = deque(maxlen=buffer_size)
resp_raw_buffer = deque(maxlen=buffer_size)
resp_filtered_buffer = deque(maxlen=buffer_size)

# Manual detection
heartbeat_taps = deque(maxlen=20)
respiration_taps = deque(maxlen=10)

# Current values
heart_rate = 0
respiration_rate = 0
manual_heart_rate = 0
manual_respiration_rate = 0

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Face ROI indices
face_roi_indices = [
    10,  103, 14, 332
]

def simple_pos_algorithm(rgb_data):
    """Simple POS algorithm"""
    if rgb_data.shape[1] < min_window:
        return 0
    
    # Normalize
    rgb_norm = rgb_data / (np.mean(rgb_data, axis=1, keepdims=True) + 1e-9)
    
    # POS projection
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    projected = np.dot(P, rgb_norm)
    S1, S2 = projected[0], projected[1]
    
    # Tuning
    alpha = np.std(S1) / (np.std(S2) + 1e-9)
    pos_signal = S1 - alpha * S2
    
    return pos_signal[-1]  # Return latest value

def bandpass_filter(data, low_freq, high_freq):
    """Simple bandpass filter"""
    if len(data) < 90:
        return data[-1] if data else 0
    
    data_array = np.array(list(data)[-90:])
    nyquist = fps / 2
    low, high = low_freq / nyquist, high_freq / nyquist
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data_array)
        return filtered[-1]
    except:
        return data_array[-1]

def apply_savgol_filter(data, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter"""
    if len(data) < window_length:
        return data[-1] if data else 0
    
    data_array = np.array(list(data)[-window_length:])
    try:
        filtered = savgol_filter(data_array, window_length, polyorder)
        return filtered[-1]
    except:
        return data_array[-1]

def estimate_heart_rate():
    """Estimate heart rate using FFT"""
    global heart_rate
    if len(pos_savgol_buffer) < 150:
        return
    
    data = np.array(list(pos_savgol_buffer)[-150:])
    data = data - np.mean(data)
    
    windowed = data * np.hanning(len(data))
    fft_vals = np.abs(np.fft.fft(windowed))
    freqs = np.fft.fftfreq(len(windowed), 1/fps)
    
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_vals[:len(fft_vals)//2]
    
    hr_mask = (pos_freqs >= 0.7) & (pos_freqs <= 4.0)
    if np.any(hr_mask):
        hr_freqs = pos_freqs[hr_mask]
        hr_power = pos_fft[hr_mask]
        peak_freq = hr_freqs[np.argmax(hr_power)]
        heart_rate = peak_freq * 60

def estimate_respiration_rate():
    """Estimate respiration rate using FFT"""
    global respiration_rate
    if len(resp_filtered_buffer) < 180:
        return
    
    data = np.array(list(resp_filtered_buffer)[-180:])
    data = data - np.mean(data)
    
    windowed = data * np.hanning(len(data))
    fft_vals = np.abs(np.fft.fft(windowed))
    freqs = np.fft.fftfreq(len(windowed), 1/fps)
    
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_vals[:len(fft_vals)//2]
    
    resp_mask = (pos_freqs >= 0.1) & (pos_freqs <= 0.8)
    if np.any(resp_mask):
        resp_freqs = pos_freqs[resp_mask]
        resp_power = pos_fft[resp_mask]
        peak_freq = resp_freqs[np.argmax(resp_power)]
        respiration_rate = peak_freq * 60

def detect_face(frame):
    """Detect face and extract RGB"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Get ROI points
        roi_points = []
        for idx in face_roi_indices:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            roi_points.append([x, y])
        
        roi_points = np.array(roi_points)
        x_min, y_min = np.max([np.min(roi_points, axis=0) - 20, [0, 0]], axis=0)
        x_max, y_max = np.min([np.max(roi_points, axis=0) + 20, [w, h]], axis=0)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        
        # Extract RGB
        face_region = frame[y_min:y_max, x_min:x_max]
        mask_region = mask[y_min:y_max, x_min:x_max]
        
        if face_region.size > 0 and mask_region.size > 0:
            face_masked = cv2.bitwise_and(face_region, face_region, mask=mask_region)
            
            b_mean = np.mean(face_masked[:, :, 0][mask_region > 0])
            g_mean = np.mean(face_masked[:, :, 1][mask_region > 0])
            r_mean = np.mean(face_masked[:, :, 2][mask_region > 0])
            
            # Draw face ROI
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Face ROI", (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return r_mean, g_mean, b_mean
    
    return None, None, None

def detect_shoulders(frame):
    """Detect shoulders"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
        mid_y = (left_y + right_y) // 2
        
        # Draw landmarks
        cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)
        cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2)
        cv2.putText(frame, f"Shoulder Y: {mid_y}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return mid_y
    
    return None

def process_frame(frame):
    """Process single frame"""
    # Face detection and RGB extraction
    r, g, b = detect_face(frame)
    if r is not None:
        rgb_buffer.append([r, g, b])
        
        # POS processing
        if len(rgb_buffer) >= min_window:
            recent_rgb = np.array(list(rgb_buffer)[-min_window:]).T
            pos_raw = simple_pos_algorithm(recent_rgb)
            pos_raw_buffer.append(pos_raw)
            
            # Apply filters
            if len(pos_raw_buffer) >= 30:
                # Bandpass filter
                pos_filtered = bandpass_filter(pos_raw_buffer, 0.7, 4.0)
                pos_filtered_buffer.append(pos_filtered)
                
                # Savitzky-Golay filter
                if len(pos_filtered_buffer) >= 11:
                    pos_savgol = apply_savgol_filter(pos_filtered_buffer, 11, 3)
                    pos_savgol_buffer.append(pos_savgol)
                    
                    # Update heart rate every second
                    if len(pos_savgol_buffer) % fps == 0:
                        estimate_heart_rate()
    
    # Shoulder detection
    shoulder_y = detect_shoulders(frame)
    if shoulder_y is not None:
        shoulder_buffer.append(shoulder_y)
        
        # Respiration processing
        if len(shoulder_buffer) >= 30:
            y_window = np.array(list(shoulder_buffer)[-30:])
            if len(y_window) > 1:
                movement = np.mean(np.diff(y_window))
                resp_raw_buffer.append(movement)
                
                # Filter respiration
                if len(resp_raw_buffer) >= 60:
                    resp_filtered = bandpass_filter(resp_raw_buffer, 0.1, 0.8)
                    resp_filtered_buffer.append(resp_filtered)
                    
                    # Update respiration rate every 2 seconds
                    if len(resp_filtered_buffer) % (fps * 2) == 0:
                        estimate_respiration_rate()

def tap_heartbeat():
    """Manual heartbeat tap"""
    global manual_heart_rate
    current_time = time.time()
    heartbeat_taps.append(current_time)
    
    if len(heartbeat_taps) >= 2:
        recent_taps = list(heartbeat_taps)[-10:]
        intervals = []
        for i in range(1, len(recent_taps)):
            interval = recent_taps[i] - recent_taps[i-1]
            if 0.3 <= interval <= 2.0:
                intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            manual_heart_rate = 60.0 / avg_interval

def tap_respiration():
    """Manual respiration tap"""
    global manual_respiration_rate
    current_time = time.time()
    respiration_taps.append(current_time)
    
    if len(respiration_taps) >= 2:
        recent_taps = list(respiration_taps)[-5:]
        intervals = []
        for i in range(1, len(recent_taps)):
            interval = recent_taps[i] - recent_taps[i-1]
            if 1.0 <= interval <= 10.0:
                intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            manual_respiration_rate = 60.0 / avg_interval

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simplified rPPG & Respiration Monitor")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side - Camera and controls
        left_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black;")
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        left_layout.addLayout(controls)
        
        # Manual detection
        manual_layout = QVBoxLayout()
        manual_title = QLabel("Manual Detection")
        manual_title.setFont(QFont("Arial", 12, QFont.Bold))
        manual_layout.addWidget(manual_title)
        
        manual_buttons = QHBoxLayout()
        self.heartbeat_btn = QPushButton("💓 Heartbeat")
        self.respiration_btn = QPushButton("🫁 Breathing")
        self.heartbeat_btn.clicked.connect(self.tap_heartbeat)
        self.respiration_btn.clicked.connect(self.tap_respiration)
        manual_buttons.addWidget(self.heartbeat_btn)
        manual_buttons.addWidget(self.respiration_btn)
        manual_layout.addLayout(manual_buttons)
        
        reset_buttons = QHBoxLayout()
        reset_hr = QPushButton("Reset HR")
        reset_rr = QPushButton("Reset RR")
        reset_hr.clicked.connect(self.reset_heartbeat)
        reset_rr.clicked.connect(self.reset_respiration)
        reset_buttons.addWidget(reset_hr)
        reset_buttons.addWidget(reset_rr)
        manual_layout.addLayout(reset_buttons)
        
        left_layout.addLayout(manual_layout)
        
        # Status
        self.status_label = QLabel("Status: Ready")
        left_layout.addWidget(self.status_label)
        
        # Right side - Plots
        right_layout = QVBoxLayout()
        
        # Info labels
        info_layout = QHBoxLayout()
        self.auto_hr_label = QLabel("Auto HR: -- BPM")
        self.auto_rr_label = QLabel("Auto RR: -- BPM")
        self.manual_hr_label = QLabel("Manual HR: -- BPM")
        self.manual_rr_label = QLabel("Manual RR: -- BPM")
        info_layout.addWidget(self.auto_hr_label)
        info_layout.addWidget(self.auto_rr_label)
        info_layout.addWidget(self.manual_hr_label)
        info_layout.addWidget(self.manual_rr_label)
        right_layout.addLayout(info_layout)
        
        # Create plots
        self.setup_plots()
        right_layout.addWidget(self.rgb_plot)
        right_layout.addWidget(self.pos_raw_plot)
        right_layout.addWidget(self.pos_final_plot)
        right_layout.addWidget(self.resp_raw_plot)
        right_layout.addWidget(self.resp_final_plot)
        
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        
        # Timer and camera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def setup_plots(self):
        """Setup all plots"""
        # RGB plot
        self.rgb_plot = pg.GraphicsLayoutWidget()
        rgb_plot = self.rgb_plot.addPlot(title="RGB Signals")
        rgb_plot.setLabel('left', 'Intensity')
        rgb_plot.setLabel('bottom', 'Time (s)')
        self.r_curve = rgb_plot.plot(pen='r', name='Red')
        self.g_curve = rgb_plot.plot(pen='g', name='Green')
        self.b_curve = rgb_plot.plot(pen='b', name='Blue')
        rgb_plot.addLegend()
        
        # POS Raw plot
        self.pos_raw_plot = pg.GraphicsLayoutWidget()
        pos_raw_plot = self.pos_raw_plot.addPlot(title="POS Raw Signal")
        pos_raw_plot.setLabel('left', 'Amplitude')
        pos_raw_plot.setLabel('bottom', 'Time (s)')
        self.pos_raw_curve = pos_raw_plot.plot(pen='orange', name='Raw POS')
        
        # POS Final plot
        self.pos_final_plot = pg.GraphicsLayoutWidget()
        pos_final_plot = self.pos_final_plot.addPlot(title="POS Final Signal (Savgol Filtered)")
        pos_final_plot.setLabel('left', 'Amplitude')
        pos_final_plot.setLabel('bottom', 'Time (s)')
        self.pos_savgol_curve = pos_final_plot.plot(pen='darkred', name='Final POS', width=3)
        
        # Respiration Raw plot
        self.resp_raw_plot = pg.GraphicsLayoutWidget()
        resp_raw_plot = self.resp_raw_plot.addPlot(title="Respiration Raw Signal")
        resp_raw_plot.setLabel('left', 'Amplitude')
        resp_raw_plot.setLabel('bottom', 'Time (s)')
        self.resp_raw_curve = resp_raw_plot.plot(pen='cyan', name='Raw Respiration')
        
        # Respiration Final plot
        self.resp_final_plot = pg.GraphicsLayoutWidget()
        resp_final_plot = self.resp_final_plot.addPlot(title="Respiration Final Signal (Filtered)")
        resp_final_plot.setLabel('left', 'Amplitude')
        resp_final_plot.setLabel('bottom', 'Time (s)')
        self.resp_filtered_curve = resp_final_plot.plot(pen='blue', name='Final Respiration')

    def start_monitoring(self):
        """Start monitoring"""
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.timer.start(33)  # ~30 FPS
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Monitoring...")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Ready")
        
    def update_frame(self):
        """Update frame and plots"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Process frame
                process_frame(frame)
                
                # Display frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled = pixmap.scaled(self.video_label.size(), aspectRatioMode=1)
                self.video_label.setPixmap(scaled)
                
                # Update plots
                self.update_plots()
                
                # Update labels
                self.auto_hr_label.setText(f"Auto HR: {heart_rate:.1f} BPM")
                self.auto_rr_label.setText(f"Auto RR: {respiration_rate:.1f} BPM")
                self.manual_hr_label.setText(f"Manual HR: {manual_heart_rate:.1f} BPM")
                self.manual_rr_label.setText(f"Manual RR: {manual_respiration_rate:.1f} BPM")
    
    def update_plots(self):
        """Update all plots"""
        # RGB signals
        if rgb_buffer:
            time_axis = np.arange(len(rgb_buffer)) / fps
            rgb_data = np.array(rgb_buffer)
            if len(rgb_data) > 0:
                self.r_curve.setData(time_axis, rgb_data[:, 0])
                self.g_curve.setData(time_axis, rgb_data[:, 1])
                self.b_curve.setData(time_axis, rgb_data[:, 2])
        
        # POS Raw signal
        if pos_raw_buffer:
            pos_time = np.arange(len(pos_raw_buffer)) / fps
            self.pos_raw_curve.setData(pos_time, list(pos_raw_buffer))
        
        # POS Final signal
        if pos_savgol_buffer:
            savgol_time = np.arange(len(pos_savgol_buffer)) / fps
            self.pos_savgol_curve.setData(savgol_time, list(pos_savgol_buffer))
        
        # Respiration Raw signal
        if resp_raw_buffer:
            resp_time = np.arange(len(resp_raw_buffer)) / fps
            self.resp_raw_curve.setData(resp_time, list(resp_raw_buffer))
            
        # Respiration Final signal
        if resp_filtered_buffer:
            resp_filt_time = np.arange(len(resp_filtered_buffer)) / fps
            self.resp_filtered_curve.setData(resp_filt_time, list(resp_filtered_buffer))
    
    def tap_heartbeat(self):
        """Handle heartbeat tap"""
        tap_heartbeat()
        
    def tap_respiration(self):
        """Handle respiration tap"""
        tap_respiration()
        
    def reset_heartbeat(self):
        """Reset heartbeat"""
        global manual_heart_rate
        heartbeat_taps.clear()
        manual_heart_rate = 0
        
    def reset_respiration(self):
        """Reset respiration"""
        global manual_respiration_rate
        respiration_taps.clear()
        manual_respiration_rate = 0
        
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())