import tkinter as tk
from tkinter import Frame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

class ShoulderRespirationGUI:
    """GUI untuk visualisasi respirasi berdasarkan pergerakan bahu"""
    
    def __init__(self):
        self.raw_movement_data = deque(maxlen=300)
        self.filtered_signal_data = deque(maxlen=300)
        self.respiration_rates = deque(maxlen=50)
        self.shoulder_positions = {'left_y': deque(maxlen=300), 'right_y': deque(maxlen=300), 'center_y': deque(maxlen=300)}
        
        self.current_bpm = 0
        self.is_recording = False
        
        self.setup_gui()
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)
        
    def setup_gui(self):
        """Setup GUI utama"""
        self.root = tk.Tk()
        self.root.title("Shoulder-based Respiration Detection")
        self.root.geometry("1000x600")  # Reduced height to fit all graphs
        self.root.configure(bg='#2c3e50')
        
        self.create_control_panel()
        self.create_plot_area()
        self.create_status_panel()
        
    def create_control_panel(self):
        """Panel kontrol"""
        control_frame = Frame(self.root, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        title_label = tk.Label(control_frame, text="Shoulder Movement Respiration Monitor", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#34495e')
        title_label.pack(pady=10)
        
        btn_frame = Frame(control_frame, bg='#34495e')
        btn_frame.pack(pady=5)
        
        self.start_btn = tk.Button(btn_frame, text="Start Detection", 
                                  command=self.toggle_detection,
                                  bg='#27ae60', fg='white', font=('Arial', 10, 'bold'),
                                  width=15, height=2)
        self.start_btn.pack(side='left', padx=5)
        
        reset_btn = tk.Button(btn_frame, text="Reset Signals", 
                             command=self.reset_signals,
                             bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                             width=15, height=2)
        reset_btn.pack(side='left', padx=5)
        
    def create_plot_area(self):
        """Area plotting dengan 4 subplot"""
        plot_frame = Frame(self.root, bg='white', relief='sunken', bd=2)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 6))  # Reduced from 12x8 to 10x6
        self.fig.patch.set_facecolor('#ecf0f1')
        
        # Plot 1: Shoulder Positions
        self.ax1.set_title('Shoulder Positions (Y-coordinates)')
        self.ax1.set_ylabel('Y Position (pixels)')
        self.ax1.grid(True)
        self.line_left, = self.ax1.plot([], [], 'g-', label='Left Shoulder', linewidth=2)
        self.line_right, = self.ax1.plot([], [], 'b-', label='Right Shoulder', linewidth=2)
        self.line_center, = self.ax1.plot([], [], 'r-', label='Center', linewidth=3)
        self.ax1.legend()
        self.ax1.invert_yaxis()  # Invert Y axis (screen coordinates)
        
        # Plot 2: Raw Movement Signal
        self.ax2.set_title('Raw Shoulder Movement Signal')
        self.ax2.set_ylabel('Movement')
        self.ax2.grid(True)
        self.line_raw, = self.ax2.plot([], [], 'orange', linewidth=2, label='Raw Signal')
        self.ax2.legend()
        
        # Plot 3: Filtered Respiration Signal
        self.ax3.set_title('Filtered Respiration Signal')
        self.ax3.set_ylabel('Amplitude')
        self.ax3.set_xlabel('Time (frames)')
        self.ax3.grid(True)
        self.line_filtered, = self.ax3.plot([], [], 'purple', linewidth=2, label='Filtered')
        self.ax3.legend()
        
        # Plot 4: Respiration Rate
        self.ax4.set_title('Respiration Rate (BPM)')
        self.ax4.set_ylabel('BPM')
        self.ax4.set_xlabel('Time (measurements)')
        self.ax4.set_ylim(8, 40)
        self.ax4.grid(True)
        self.line_bpm, = self.ax4.plot([], [], 'red', linewidth=3, marker='o', 
                                      markersize=4, label='BPM')
        self.ax4.legend()
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_status_panel(self):
        """Panel status"""
        status_frame = Frame(self.root, bg='#34495e', relief='raised', bd=2)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        indicators_frame = Frame(status_frame, bg='#34495e')
        indicators_frame.pack(fill='x', pady=5)
        
        self.bpm_label = tk.Label(indicators_frame, text="Respiration: -- BPM", 
                                 font=('Arial', 14, 'bold'), fg='#e74c3c', bg='#34495e')
        self.bpm_label.pack(side='left', padx=20)
        
        self.quality_label = tk.Label(indicators_frame, text="Signal: Initializing", 
                                     font=('Arial', 12), fg='#f39c12', bg='#34495e')
        self.quality_label.pack(side='left', padx=20)
        
        self.recording_label = tk.Label(indicators_frame, text="● READY", 
                                       font=('Arial', 12, 'bold'), fg='#95a5a6', bg='#34495e')
        self.recording_label.pack(side='right', padx=20)
        
    def update_data(self, raw_signal=None, filtered_signal=None, respiration_rate=None, shoulder_data=None):
        """Update data buffers"""
        if shoulder_data is not None:
            self.shoulder_positions['left_y'].append(shoulder_data.get('left_y', 0))
            self.shoulder_positions['right_y'].append(shoulder_data.get('right_y', 0))
            self.shoulder_positions['center_y'].append(shoulder_data.get('center_y', 0))
            
        if raw_signal is not None:
            self.raw_movement_data.append(raw_signal)
            
        if filtered_signal is not None:
            self.filtered_signal_data.extend(filtered_signal[-50:] if len(filtered_signal) > 50 else filtered_signal)
            
        if respiration_rate is not None and respiration_rate > 0:
            self.respiration_rates.append(respiration_rate)
            self.current_bpm = respiration_rate
            
    def update_plots(self, frame):
        """Update semua plot secara real-time"""
        # Update shoulder positions
        if len(self.shoulder_positions['center_y']) > 0:
            x_pos = range(len(self.shoulder_positions['center_y']))
            self.line_left.set_data(x_pos, list(self.shoulder_positions['left_y']))
            self.line_right.set_data(x_pos, list(self.shoulder_positions['right_y']))
            self.line_center.set_data(x_pos, list(self.shoulder_positions['center_y']))
            self.ax1.relim()
            self.ax1.autoscale_view()
            
        # Update raw movement
        if len(self.raw_movement_data) > 0:
            x_raw = range(len(self.raw_movement_data))
            self.line_raw.set_data(x_raw, list(self.raw_movement_data))
            self.ax2.relim()
            self.ax2.autoscale_view()
            
        # Update filtered signal
        if len(self.filtered_signal_data) > 0:
            x_filtered = range(len(self.filtered_signal_data))
            self.line_filtered.set_data(x_filtered, list(self.filtered_signal_data))
            self.ax3.relim()
            self.ax3.autoscale_view()
            
        # Update BPM
        if len(self.respiration_rates) > 0:
            x_bpm = range(len(self.respiration_rates))
            self.line_bpm.set_data(x_bpm, list(self.respiration_rates))
            self.ax4.relim()
            self.ax4.autoscale_view()
            
        self.update_status_labels()
        
        return [self.line_left, self.line_right, self.line_center, self.line_raw, 
                self.line_filtered, self.line_bpm]
    
    def update_status_labels(self):
        """Update label status"""
        if self.current_bpm > 0:
            self.bpm_label.config(text=f"Respiration: {self.current_bpm:.1f} BPM", fg='#27ae60')
        else:
            self.bpm_label.config(text="Respiration: -- BPM", fg='#e74c3c')
            
        data_length = len(self.raw_movement_data)
        if data_length > 200:
            self.quality_label.config(text="Signal: Excellent", fg='#27ae60')
        elif data_length > 100:
            self.quality_label.config(text="Signal: Good", fg='#f39c12')
        else:
            self.quality_label.config(text="Signal: Initializing", fg='#e74c3c')
        
        if self.is_recording:
            self.recording_label.config(text="● RECORDING", fg='#e74c3c')
        else:
            self.recording_label.config(text="● READY", fg='#95a5a6')
    
    def toggle_detection(self):
        """Toggle start/stop detection"""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.start_btn.config(text="Stop Detection", bg='#e74c3c')
        else:
            self.start_btn.config(text="Start Detection", bg='#27ae60')
    
    def reset_signals(self):
        """Reset semua buffer sinyal"""
        self.raw_movement_data.clear()
        self.filtered_signal_data.clear()
        self.respiration_rates.clear()
        for channel in self.shoulder_positions.values():
            channel.clear()
        self.current_bpm = 0
    
    def run(self):
        """Jalankan GUI"""
        self.root.mainloop()
        
    def close(self):
        """Tutup GUI"""
        self.root.quit()
        self.root.destroy()