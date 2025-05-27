import tkinter as tk
from tkinter import Frame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

class RealtimeSignalGUI:
    """GUI untuk visualisasi real-time sinyal rPPG heart rate"""
    
    def __init__(self):
        self.raw_signal_data = deque(maxlen=150)
        self.processed_signal_data = deque(maxlen=150)
        self.heart_rates = deque(maxlen=50)
        self.rgb_signals = {'R': deque(maxlen=150), 'G': deque(maxlen=150), 'B': deque(maxlen=150)}
        
        self.current_bpm = 0
        self.is_recording = False
        
        self.setup_gui()
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)
        
    def setup_gui(self):
        """Setup GUI utama"""
        self.root = tk.Tk()
        self.root.title("Real-time Heart Rate Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        self.create_control_panel()
        self.create_plot_area()
        self.create_status_panel()
        
    def create_control_panel(self):
        """Panel kontrol"""
        control_frame = Frame(self.root, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        title_label = tk.Label(control_frame, text="rPPG Heart Rate Monitor", 
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
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor('#ecf0f1')
        
        self.ax1.set_title('Raw RGB Signals')
        self.line_r, = self.ax1.plot([], [], 'r-', label='Red')
        self.line_g, = self.ax1.plot([], [], 'g-', label='Green')
        self.line_b, = self.ax1.plot([], [], 'b-', label='Blue')
        self.ax1.legend()
        self.ax1.grid(True)
        
        self.ax2.set_title('Raw rPPG Signal (POS)')
        self.line_raw, = self.ax2.plot([], [], 'purple', label='Raw Signal')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.ax3.set_title('Filtered Heart Rate Signal')
        self.line_filtered, = self.ax3.plot([], [], 'orange', label='Filtered')
        self.ax3.legend()
        self.ax3.grid(True)
        
        self.ax4.set_title('Heart Rate (BPM)')
        self.ax4.set_ylim(40, 180)
        self.line_bpm, = self.ax4.plot([], [], 'red', linewidth=3, marker='o', label='BPM')
        self.ax4.legend()
        self.ax4.grid(True)
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_status_panel(self):
        """Panel status"""
        status_frame = Frame(self.root, bg='#34495e', relief='raised', bd=2)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        indicators_frame = Frame(status_frame, bg='#34495e')
        indicators_frame.pack(fill='x', pady=5)
        
        self.bpm_label = tk.Label(indicators_frame, text="Heart Rate: -- BPM", 
                                 font=('Arial', 14, 'bold'), fg='#e74c3c', bg='#34495e')
        self.bpm_label.pack(side='left', padx=20)
        
        self.quality_label = tk.Label(indicators_frame, text="Signal: Initializing", 
                                     font=('Arial', 12), fg='#f39c12', bg='#34495e')
        self.quality_label.pack(side='left', padx=20)
        
        self.recording_label = tk.Label(indicators_frame, text="● READY", 
                                       font=('Arial', 12, 'bold'), fg='#95a5a6', bg='#34495e')
        self.recording_label.pack(side='right', padx=20)
        
    def update_data(self, raw_signal=None, processed_signal=None, heart_rate=None, rgb_values=None):
        """Update data buffers"""
        if rgb_values is not None:
            self.rgb_signals['R'].append(rgb_values[0])
            self.rgb_signals['G'].append(rgb_values[1])
            self.rgb_signals['B'].append(rgb_values[2])
            
        if raw_signal is not None:
            self.raw_signal_data.append(raw_signal)
            
        if processed_signal is not None:
            self.processed_signal_data.append(processed_signal)
            
        if heart_rate is not None and heart_rate > 0:
            self.heart_rates.append(heart_rate)
            self.current_bpm = heart_rate
            
    def update_plots(self, frame):
        """Update semua plot secara real-time"""
        if len(self.rgb_signals['R']) > 0:
            x_rgb = range(len(self.rgb_signals['R']))
            self.line_r.set_data(x_rgb, list(self.rgb_signals['R']))
            self.line_g.set_data(x_rgb, list(self.rgb_signals['G']))
            self.line_b.set_data(x_rgb, list(self.rgb_signals['B']))
            self.ax1.relim()
            self.ax1.autoscale_view()
            
        if len(self.raw_signal_data) > 0:
            x_raw = range(len(self.raw_signal_data))
            self.line_raw.set_data(x_raw, list(self.raw_signal_data))
            self.ax2.relim()
            self.ax2.autoscale_view()
            
        if len(self.processed_signal_data) > 0:
            x_filtered = range(len(self.processed_signal_data))
            self.line_filtered.set_data(x_filtered, list(self.processed_signal_data))
            self.ax3.relim()
            self.ax3.autoscale_view()
            
        if len(self.heart_rates) > 0:
            x_bpm = range(len(self.heart_rates))
            self.line_bpm.set_data(x_bpm, list(self.heart_rates))
            self.ax4.relim()
            self.ax4.autoscale_view()
            
        self.update_status_labels()
        
        return [self.line_r, self.line_g, self.line_b, self.line_raw, 
                self.line_filtered, self.line_bpm]
    
    def update_status_labels(self):
        """Update label status"""
        if self.current_bpm > 0:
            self.bpm_label.config(text=f"Heart Rate: {self.current_bpm:.1f} BPM", fg='#27ae60')
        else:
            self.bpm_label.config(text="Heart Rate: -- BPM", fg='#e74c3c')
            
        data_length = len(self.raw_signal_data)
        if data_length > 100:
            self.quality_label.config(text="Signal: Excellent", fg='#27ae60')
        elif data_length > 50:
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
        self.raw_signal_data.clear()
        self.processed_signal_data.clear()
        self.heart_rates.clear()
        for channel in self.rgb_signals.values():
            channel.clear()
        self.current_bpm = 0
    
    def run(self):
        """Jalankan GUI"""
        self.root.mainloop()
        
    def close(self):
        """Tutup GUI"""
        self.root.quit()
        self.root.destroy()