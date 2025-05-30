import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class SignalVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time rPPG and Respiration Monitor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set up the layout
        main_layout = QVBoxLayout()
        
        # Title and info labels
        title_label = QLabel("Real-time rPPG Heart Rate & Respiration Monitor")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Info layout
        info_layout = QHBoxLayout()
        
        # Automatic detection labels
        auto_label = QLabel("Automatic Detection:")
        auto_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.heart_rate_label = QLabel("Heart Rate: -- BPM")
        self.heart_rate_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.heart_rate_label.setStyleSheet("color: red;")
        
        self.respiration_rate_label = QLabel("Respiration Rate: -- BPM")
        self.respiration_rate_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.respiration_rate_label.setStyleSheet("color: blue;")
        
        # Manual detection labels
        manual_label = QLabel("Manual Detection:")
        manual_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.manual_heart_rate_label = QLabel("Manual HR: -- BPM")
        self.manual_heart_rate_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.manual_heart_rate_label.setStyleSheet("color: darkred;")
        
        self.manual_respiration_rate_label = QLabel("Manual RR: -- BPM")
        self.manual_respiration_rate_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.manual_respiration_rate_label.setStyleSheet("color: darkblue;")
        
        info_layout.addWidget(auto_label)
        info_layout.addWidget(self.heart_rate_label)
        info_layout.addWidget(self.respiration_rate_label)
        info_layout.addStretch()
        info_layout.addWidget(manual_label)
        info_layout.addWidget(self.manual_heart_rate_label)
        info_layout.addWidget(self.manual_respiration_rate_label)
        main_layout.addLayout(info_layout)
        
        # Create plots
        self.setup_plots()
        
        # Add plots to layout
        plots_layout = QVBoxLayout()
        plots_layout.addWidget(self.rgb_plot_widget)
        plots_layout.addWidget(self.pos_plot_widget)
        plots_layout.addWidget(self.respiration_plot_widget)
        
        main_layout.addLayout(plots_layout)
        self.setLayout(main_layout)
        
        # Data storage
        self.max_points = 300  # 10 seconds at 30fps
        self.time_axis = np.arange(self.max_points) / 30.0  # Time in seconds
        
    def setup_plots(self):
        """Set up all the plot widgets"""
        
        # RGB Signals Plot
        self.rgb_plot_widget = pg.GraphicsLayoutWidget()
        self.rgb_plot = self.rgb_plot_widget.addPlot(title="RGB Face Signals")
        self.rgb_plot.setLabel('left', 'Intensity')
        self.rgb_plot.setLabel('bottom', 'Time (s)')
        self.rgb_plot.showGrid(x=True, y=True)
        
        # RGB curves
        self.red_curve = self.rgb_plot.plot(pen=pg.mkPen('r', width=2), name='Red')
        self.green_curve = self.rgb_plot.plot(pen=pg.mkPen('g', width=2), name='Green')
        self.blue_curve = self.rgb_plot.plot(pen=pg.mkPen('b', width=2), name='Blue')
        self.rgb_plot.addLegend()
        
        # POS Signals Plot - Side by side
        self.pos_plot_widget = pg.GraphicsLayoutWidget()
        
        # POS Raw Plot (left side)
        self.pos_raw_plot = self.pos_plot_widget.addPlot(title="POS Raw Signal")
        self.pos_raw_plot.setLabel('left', 'Amplitude')
        self.pos_raw_plot.setLabel('bottom', 'Time (s)')
        self.pos_raw_plot.showGrid(x=True, y=True)
        self.pos_raw_curve = self.pos_raw_plot.plot(pen=pg.mkPen('orange', width=2))
        
        # POS Filtered Plot (right side)
        self.pos_filtered_plot = self.pos_plot_widget.addPlot(title="POS Filtered Signal")
        self.pos_filtered_plot.setLabel('left', 'Amplitude')
        self.pos_filtered_plot.setLabel('bottom', 'Time (s)')
        self.pos_filtered_plot.showGrid(x=True, y=True)
        self.pos_filtered_curve = self.pos_filtered_plot.plot(pen=pg.mkPen('red', width=3))
        
        # Respiration Signals Plot
        self.respiration_plot_widget = pg.GraphicsLayoutWidget()
        self.respiration_plot = self.respiration_plot_widget.addPlot(title="Respiration Signals")
        self.respiration_plot.setLabel('left', 'Amplitude')
        self.respiration_plot.setLabel('bottom', 'Time (s)')
        self.respiration_plot.showGrid(x=True, y=True)
        
        # Respiration curves
        self.resp_raw_curve = self.respiration_plot.plot(pen=pg.mkPen('cyan', width=2), name='Respiration Raw')
        self.resp_filtered_curve = self.respiration_plot.plot(pen=pg.mkPen('blue', width=3), name='Respiration Filtered')
        self.respiration_plot.addLegend()
    
    def update_rgb_signals(self, rgb_data):
        """Update RGB signals plot"""
        if rgb_data:
            red_data = rgb_data.get('red', [])
            green_data = rgb_data.get('green', [])
            blue_data = rgb_data.get('blue', [])
            
            if len(red_data) > 0:
                # Pad or truncate data to match time axis
                red_padded = self._pad_or_truncate(red_data)
                green_padded = self._pad_or_truncate(green_data)
                blue_padded = self._pad_or_truncate(blue_data)
                
                # Update curves
                self.red_curve.setData(self.time_axis[:len(red_padded)], red_padded)
                self.green_curve.setData(self.time_axis[:len(green_padded)], green_padded)
                self.blue_curve.setData(self.time_axis[:len(blue_padded)], blue_padded)
    
    def update_pos_signals(self, pos_data):
        """Update POS signals plot"""
        if pos_data:
            raw_data = pos_data.get('raw', [])
            filtered_data = pos_data.get('filtered', [])
            
            if len(raw_data) > 0:
                raw_padded = self._pad_or_truncate(raw_data)
                self.pos_raw_curve.setData(self.time_axis[:len(raw_padded)], raw_padded)
            
            if len(filtered_data) > 0:
                filtered_padded = self._pad_or_truncate(filtered_data)
                self.pos_filtered_curve.setData(self.time_axis[:len(filtered_padded)], filtered_padded)
    
    def update_respiration_signals(self, resp_data):
        """Update respiration signals plot"""
        if resp_data:
            raw_data = resp_data.get('raw', [])
            filtered_data = resp_data.get('filtered', [])
            
            if len(raw_data) > 0:
                raw_padded = self._pad_or_truncate(raw_data)
                self.resp_raw_curve.setData(self.time_axis[:len(raw_padded)], raw_padded)
            
            if len(filtered_data) > 0:
                filtered_padded = self._pad_or_truncate(filtered_data)
                self.resp_filtered_curve.setData(self.time_axis[:len(filtered_padded)], filtered_padded)
    
    def update_vital_signs(self, heart_rate, respiration_rate):
        """Update vital signs display"""
        self.heart_rate_label.setText(f"Heart Rate: {heart_rate:.1f} BPM")
        self.respiration_rate_label.setText(f"Respiration Rate: {respiration_rate:.1f} BPM")
    
    def update_manual_vital_signs(self, manual_heart_rate, manual_respiration_rate):
        """Update manual vital signs display"""
        self.manual_heart_rate_label.setText(f"Manual HR: {manual_heart_rate:.1f} BPM")
        self.manual_respiration_rate_label.setText(f"Manual RR: {manual_respiration_rate:.1f} BPM")
    
    def _pad_or_truncate(self, data):
        """Pad or truncate data to fit the display window"""
        data_array = np.array(data)
        if len(data_array) > self.max_points:
            return data_array[-self.max_points:]
        elif len(data_array) < self.max_points:
            # Pad with zeros at the beginning
            padding = np.zeros(self.max_points - len(data_array))
            return np.concatenate([padding, data_array])
        return data_array