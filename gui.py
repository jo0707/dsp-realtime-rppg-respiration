import sys
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QPushButton, QLabel, QGridLayout, QGroupBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

class MainWindow(QMainWindow):
    def __init__(self, camera_processor, signal_processor):
        super().__init__()
        self.camera_processor = camera_processor
        self.signal_processor = signal_processor

        # set judul dan ukuran window
        self.setWindowTitle("Simplified rPPG & Respiration Monitor")
        self.setGeometry(50, 50, 1600, 1000)
        self.setStyleSheet("""
            QWidget {
                background-color: #fafafa;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                background-color: #fafafa;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 20px;
                padding-top: 10px;
                font-size: 18px;
                font-weight: bold;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                padding: 10px 20px;
                border-radius: 8px;
            }
            
            QPushButton:disabled {
                background-color: #cfd8dc;
                color: #9e9e9e;
            }

        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # layout Kiri
        left_layout = QVBoxLayout()

        # Video Display
        self.video_label = QLabel("Live Camera")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #ccc; background-color: #000; color: white;")
        left_layout.addWidget(self.video_label)

        # Tombol Start/Stop
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("""
            QPushButton {
            background-color: #4caf50;
            color: white;
            }
            QPushButton:disabled {
            background-color: #cfd8dc;
            }
            """)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
            background-color: #f44336;
            color: white;
            }
            QPushButton:disabled {
            background-color: #cfd8dc;
            }
            """)
        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        left_layout.addLayout(control_layout)

        # Hasil Auto Detection
        auto_group = QGroupBox("Auto Detection")
        auto_layout = QHBoxLayout()
        self.auto_hr_display = QLabel("--\nBPM\nHeartbeat")
        self.auto_hr_display.setAlignment(Qt.AlignCenter)
        self.auto_hr_display.setStyleSheet("background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px; padding: 15px;")
        self.auto_rr_display = QLabel("--\nBPM\nBreathing")
        self.auto_rr_display.setAlignment(Qt.AlignCenter)
        self.auto_rr_display.setStyleSheet("background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px; padding: 15px;")
        auto_layout.addWidget(self.auto_hr_display)
        auto_layout.addWidget(self.auto_rr_display)
        auto_group.setLayout(auto_layout)
        left_layout.addWidget(auto_group)

        # Hasil Manual Detection
        manual_group = QGroupBox("Manual Detection")
        manual_layout = QVBoxLayout()
        manual_value_layout = QHBoxLayout()
        self.manual_hr_display = QLabel("--\nBPM\nHeartbeat")
        self.manual_hr_display.setAlignment(Qt.AlignCenter)
        self.manual_hr_display.setStyleSheet("background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px; padding: 15px;")
        self.manual_rr_display = QLabel("--\nBPM\nBreathing")
        self.manual_rr_display.setAlignment(Qt.AlignCenter)
        self.manual_rr_display.setStyleSheet("background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px; padding: 15px;")
        manual_value_layout.addWidget(self.manual_hr_display)
        manual_value_layout.addWidget(self.manual_rr_display)

        # Tombol Manual Detection
        manual_buttons_layout = QHBoxLayout()
        self.heartbeat_btn = QPushButton("💓 Heartbeat")
        self.heartbeat_btn.setStyleSheet("background-color: #2196f3; color: white;")
        self.heartbeat_btn.clicked.connect(self.tap_heartbeat)
        self.respiration_btn = QPushButton("🫁 Breathing")
        self.respiration_btn.setStyleSheet("background-color: #2196f3; color: white;")
        self.respiration_btn.clicked.connect(self.tap_respiration)
        manual_buttons_layout.addWidget(self.heartbeat_btn)
        manual_buttons_layout.addWidget(self.respiration_btn)

        # Tombol Reset
        reset_buttons_layout = QHBoxLayout()
        self.reset_hr_btn = QPushButton("Reset HR")
        self.reset_hr_btn.setStyleSheet("background-color: #e8ebea;")
        self.reset_hr_btn.clicked.connect(self.reset_heartbeat)
        self.reset_rr_btn = QPushButton("Reset RR")
        self.reset_rr_btn.setStyleSheet("background-color: #e8ebea;")
        self.reset_rr_btn.clicked.connect(self.reset_respiration)
        reset_buttons_layout.addWidget(self.reset_hr_btn)
        reset_buttons_layout.addWidget(self.reset_rr_btn)

        # Gabungkan semua layout manual
        manual_layout.addLayout(manual_value_layout)
        manual_layout.addLayout(manual_buttons_layout)
        manual_layout.addLayout(reset_buttons_layout)
        manual_group.setLayout(manual_layout)
        left_layout.addWidget(manual_group)

        # Status sistem
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.status_label.setStyleSheet("font-style: italic; color: #666;")
        left_layout.addWidget(self.status_label)

        # Layout Kanan
        right_layout = QVBoxLayout()
        self.setup_plots()
        right_layout.addWidget(self.rgb_plot)
        right_layout.addWidget(self.pos_raw_plot)
        right_layout.addWidget(self.pos_final_plot)
        right_layout.addWidget(self.resp_raw_plot)
        right_layout.addWidget(self.resp_final_plot)
        # Main Layout
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # Setup semua plot
    def setup_plots(self):
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
        self.cap = cv2.VideoCapture(self.camera_processor.camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_processor.fps)
        self.timer.start(33)  # ~30 FPS
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #a5d6a7; color: white; transition: all 0.3s ease;")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.status_label.setText("Status: Monitoring...")
            
    def stop_monitoring(self):
        """Stop monitoring"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.start_btn.setEnabled(True)
        self.start_btn.setStyleSheet("background-color: #4caf50; color: white;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #ef9a9a; color: white; transition: all 0.3s ease;")
        self.status_label.setText("Status: Ready")

    # Perbarui frame video dan data sinyal
    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.camera_processor.process_frame(frame, self.signal_processor)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), aspectRatioMode=1))
                self.update_plots()
                self.auto_hr_display.setText(f"{self.signal_processor.heart_rate:.0f}\nBPM\nHeartbeat")
                self.auto_rr_display.setText(f"{self.signal_processor.respiration_rate:.0f}\nBPM\nBreathing")
                self.manual_hr_display.setText(f"{self.signal_processor.manual_heart_rate:.0f}\nBPM\nHeartbeat")
                self.manual_rr_display.setText(f"{self.signal_processor.manual_respiration_rate:.0f}\nBPM\nBreathing")

    # Update semua plot sinyal
    def update_plots(self):
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
    
    # Fungsi untuk mencatat detak jantung dan pernapasan secara manual
    def tap_heartbeat(self):
        self.signal_processor.tap_heartbeat()

    def tap_respiration(self):
        self.signal_processor.tap_respiration()

    def reset_heartbeat(self):
        self.signal_processor.reset_heartbeat()

    def reset_respiration(self):
        self.signal_processor.reset_respiration()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()
