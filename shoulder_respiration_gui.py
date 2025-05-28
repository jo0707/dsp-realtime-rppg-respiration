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
from manual_detector import ManualDetector

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
        
        # Manual detection section
        manual_section = QVBoxLayout()
        manual_title = QLabel("Manual Detection")
        manual_title.setFont(QFont("Arial", 12, QFont.Bold))
        manual_title.setStyleSheet("color: darkgreen;")
        manual_section.addWidget(manual_title)
        
        # Manual detection buttons
        manual_buttons_layout = QHBoxLayout()
        
        # Heartbeat tap button
        self.heartbeat_button = QPushButton("💓 Tap Heartbeat")
        self.heartbeat_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.heartbeat_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #e53935;
            }
        """)
        self.heartbeat_button.clicked.connect(self.tap_heartbeat)
        
        # Respiration tap button
        self.respiration_button = QPushButton("🫁 Tap Breathing")
        self.respiration_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.respiration_button.setStyleSheet("""
            QPushButton {
                background-color: #42a5f5;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2196f3;
            }
            QPushButton:pressed {
                background-color: #1976d2;
            }
        """)
        self.respiration_button.clicked.connect(self.tap_respiration)
        
        manual_buttons_layout.addWidget(self.heartbeat_button)
        manual_buttons_layout.addWidget(self.respiration_button)
        manual_section.addLayout(manual_buttons_layout)
        
        # Reset buttons
        reset_buttons_layout = QHBoxLayout()
        self.reset_hr_button = QPushButton("Reset HR")
        self.reset_hr_button.clicked.connect(self.reset_heartbeat)
        self.reset_rr_button = QPushButton("Reset RR")
        self.reset_rr_button.clicked.connect(self.reset_respiration)
        
        reset_buttons_layout.addWidget(self.reset_hr_button)
        reset_buttons_layout.addWidget(self.reset_rr_button)
        manual_section.addLayout(reset_buttons_layout)
        
        # Instructions
        instructions = QLabel("Instructions:\n• Tap 'Heartbeat' with each heartbeat\n• Tap 'Breathing' with each breath cycle")
        instructions.setFont(QFont("Arial", 9))
        instructions.setStyleSheet("color: gray;")
        manual_section.addWidget(instructions)
        
        left_layout.addLayout(manual_section)
        
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
        
        # Manual detector
        self.manual_detector = ManualDetector()
        
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
            
            # Update manual vital signs
            manual_hr = self.manual_detector.get_manual_heart_rate()
            manual_rr = self.manual_detector.get_manual_respiration_rate()
            self.signal_visualizer.update_manual_vital_signs(manual_hr, manual_rr)
    
    def tap_heartbeat(self):
        """Handle manual heartbeat tap"""
        self.manual_detector.tap_heartbeat()
        # Visual feedback
        self.heartbeat_button.setStyleSheet("""
            QPushButton {
                background-color: #e53935;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        # Reset button color after 100ms
        QTimer.singleShot(100, lambda: self.heartbeat_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #e53935;
            }
        """))
    
    def tap_respiration(self):
        """Handle manual respiration tap"""
        self.manual_detector.tap_respiration()
        # Visual feedback
        self.respiration_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        # Reset button color after 100ms
        QTimer.singleShot(100, lambda: self.respiration_button.setStyleSheet("""
            QPushButton {
                background-color: #42a5f5;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2196f3;
            }
            QPushButton:pressed {
                background-color: #1976d2;
            }
        """))
    
    def reset_heartbeat(self):
        """Reset manual heartbeat detection"""
        self.manual_detector.reset_heartbeat()
    
    def reset_respiration(self):
        """Reset manual respiration detection"""
        self.manual_detector.reset_respiration()
    
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