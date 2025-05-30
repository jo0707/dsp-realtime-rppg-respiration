import sys
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

class MainWindow(QMainWindow):
    def __init__(self, camera_processor, signal_processor):
        super().__init__()
        self.camera_processor = camera_processor
        self.signal_processor = signal_processor
        
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
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_processor.fps)
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
                processed_frame = self.camera_processor.process_frame(frame, self.signal_processor)
                
                # Display frame
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled = pixmap.scaled(self.video_label.size(), aspectRatioMode=1)
                self.video_label.setPixmap(scaled)
                
                # Update plots
                self.update_plots()
                
                # Update labels
                self.auto_hr_label.setText(f"Auto HR: {self.signal_processor.heart_rate:.1f} BPM")
                self.auto_rr_label.setText(f"Auto RR: {self.signal_processor.respiration_rate:.1f} BPM")
                self.manual_hr_label.setText(f"Manual HR: {self.signal_processor.manual_heart_rate:.1f} BPM")
                self.manual_rr_label.setText(f"Manual RR: {self.signal_processor.manual_respiration_rate:.1f} BPM")
    
    def update_plots(self):
        """Update all plots"""
        # RGB signals
        if self.signal_processor.rgb_buffer:
            time_axis = np.arange(len(self.signal_processor.rgb_buffer)) / self.signal_processor.fps
            rgb_data = np.array(self.signal_processor.rgb_buffer)
            if len(rgb_data) > 0:
                self.r_curve.setData(time_axis, rgb_data[:, 0])
                self.g_curve.setData(time_axis, rgb_data[:, 1])
                self.b_curve.setData(time_axis, rgb_data[:, 2])
        
        # POS Raw signal
        if self.signal_processor.pos_raw_buffer:
            pos_time = np.arange(len(self.signal_processor.pos_raw_buffer)) / self.signal_processor.fps
            self.pos_raw_curve.setData(pos_time, list(self.signal_processor.pos_raw_buffer))
        
        # POS Final signal
        if self.signal_processor.pos_savgol_buffer:
            savgol_time = np.arange(len(self.signal_processor.pos_savgol_buffer)) / self.signal_processor.fps
            self.pos_savgol_curve.setData(savgol_time, list(self.signal_processor.pos_savgol_buffer))
        
        # Respiration Raw signal
        if self.signal_processor.resp_raw_buffer:
            resp_time = np.arange(len(self.signal_processor.resp_raw_buffer)) / self.signal_processor.fps
            self.resp_raw_curve.setData(resp_time, list(self.signal_processor.resp_raw_buffer))
            
        # Respiration Final signal
        if self.signal_processor.resp_filtered_buffer:
            resp_filt_time = np.arange(len(self.signal_processor.resp_filtered_buffer)) / self.signal_processor.fps
            self.resp_filtered_curve.setData(resp_filt_time, list(self.signal_processor.resp_filtered_buffer))
    
    def tap_heartbeat(self):
        """Handle heartbeat tap"""
        self.signal_processor.tap_heartbeat()
        
    def tap_respiration(self):
        """Handle respiration tap"""
        self.signal_processor.tap_respiration()
        
    def reset_heartbeat(self):
        """Reset heartbeat"""
        self.signal_processor.reset_heartbeat()
        
    def reset_respiration(self):
        """Reset respiration"""
        self.signal_processor.reset_respiration()
        
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()