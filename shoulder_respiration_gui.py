import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import time

from camera import Camera
from face_detector import FaceDetector
from pos import POSProcessor
from shoulder_detector import ShoulderDetector
from respiration_processor import RespirationProcessor
from signal_visualizer import SignalVisualizer

class VideoProcessor(QThread):
    """Thread for processing video frames"""
    frame_processed = pyqtSignal(np.ndarray, dict, dict, float, float)
    
    def __init__(self):
        super().__init__()
        self.camera = Camera(fps=30)
        self.face_detector = FaceDetector()
        self.pos_processor = POSProcessor(fps=30)
        self.shoulder_detector = ShoulderDetector()
        self.respiration_processor = RespirationProcessor(fps=30)
        
        self.running = False
        
    def start_processing(self):
        self.running = True
        self.camera.start()
        self.start()
        
    def stop_processing(self):
        self.running = False
        self.camera.stop()
        self.quit()
        self.wait()
        
    def run(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is not None:
                # Process face detection and RGB extraction
                face_data = self.face_detector.detect_face_roi(frame)
                r, g, b = self.face_detector.extract_rgb_signals(frame, face_data)
                
                # Process rPPG
                self.pos_processor.add_rgb_sample(r, g, b)
                
                # Process shoulder detection
                shoulder_data = self.shoulder_detector.detect_shoulders(frame)
                shoulder_y = None
                if shoulder_data:
                    shoulder_y = shoulder_data['midpoint'][1]
                    frame = self.shoulder_detector.draw_landmarks(frame, shoulder_data)
                
                # Process respiration
                self.respiration_processor.add_shoulder_sample(shoulder_y)
                
                # Draw face ROI
                if face_data:
                    x_min, y_min, x_max, y_max = face_data['bbox']
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, "Face ROI", (x_min, y_min-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Get current data
                rgb_signals = self.pos_processor.get_rgb_signals()
                pos_signals = self.pos_processor.get_pos_signals()
                resp_signals = self.respiration_processor.get_respiration_signals()
                
                heart_rate = self.pos_processor.get_heart_rate()
                respiration_rate = self.respiration_processor.get_respiration_rate()
                
                # Emit processed data
                self.frame_processed.emit(
                    frame, 
                    {'rgb': rgb_signals, 'pos': pos_signals, 'resp': resp_signals},
                    shoulder_data or {},
                    heart_rate,
                    respiration_rate
                )
            
            self.msleep(33)  # ~30 FPS

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time rPPG Heart Rate & Respiration Monitor")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side - Video and controls
        left_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black;")
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        left_layout.addLayout(controls_layout)
        
        # Status labels
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 10))
        left_layout.addWidget(self.status_label)
        
        # Right side - Signal plots
        self.signal_visualizer = SignalVisualizer()
        
        # Add to main layout
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.signal_visualizer, 2)
        
        # Video processor
        self.video_processor = VideoProcessor()
        self.video_processor.frame_processed.connect(self.update_display)
        
        # Update timer for plots
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        
        # Data storage
        self.current_signals = {}
        self.current_heart_rate = 0
        self.current_respiration_rate = 0
        
    def start_monitoring(self):
        """Start the monitoring process"""
        self.status_label.setText("Status: Starting...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Start video processing
        self.video_processor.start_processing()
        
        # Start plot updates
        self.plot_timer.start(100)  # Update plots every 100ms
        
        self.status_label.setText("Status: Monitoring...")
        
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.status_label.setText("Status: Stopping...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Stop video processing
        self.video_processor.stop_processing()
        
        # Stop plot updates
        self.plot_timer.stop()
        
        self.status_label.setText("Status: Ready")
        
    def update_display(self, frame, signals, shoulder_data, heart_rate, respiration_rate):
        """Update video display and store signal data"""
        # Convert frame to Qt format and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(self.video_label.size(), aspectRatioMode=1)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Store current data
        self.current_signals = signals
        self.current_heart_rate = heart_rate
        self.current_respiration_rate = respiration_rate
        
    def update_plots(self):
        """Update signal plots"""
        if self.current_signals:
            # Update RGB signals
            if 'rgb' in self.current_signals:
                self.signal_visualizer.update_rgb_signals(self.current_signals['rgb'])
            
            # Update POS signals
            if 'pos' in self.current_signals:
                self.signal_visualizer.update_pos_signals(self.current_signals['pos'])
            
            # Update respiration signals
            if 'resp' in self.current_signals:
                self.signal_visualizer.update_respiration_signals(self.current_signals['resp'])
            
            # Update vital signs display
            self.signal_visualizer.update_vital_signs(
                self.current_heart_rate, 
                self.current_respiration_rate
            )
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.video_processor.running:
            self.stop_monitoring()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()