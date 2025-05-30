import numpy as np
from scipy import signal
from collections import deque

class RespirationProcessor:
    def __init__(self, fps=30, window_size=300):
        self.fps = fps
        self.window_size = window_size  # 10 seconds at 30fps
        
        # Signal buffers
        self.shoulder_y_buffer = deque(maxlen=window_size)
        self.respiration_raw_buffer = deque(maxlen=window_size)
        self.respiration_filtered_buffer = deque(maxlen=window_size)
        
        # Respiration rate estimation
        self.respiration_rate = 0
        self.last_rr_update = 0
        
    def add_shoulder_sample(self, shoulder_y):
        """Add new shoulder Y position to buffer"""
        if shoulder_y is not None:
            self.shoulder_y_buffer.append(shoulder_y)
            
            # Process respiration signal if we have enough samples
            if len(self.shoulder_y_buffer) >= 30:  # At least 1 second of data
                resp_raw = self._calculate_respiration_raw()
                self.respiration_raw_buffer.append(resp_raw)
                
                # Filter the signal
                if len(self.respiration_raw_buffer) >= 30:
                    resp_filtered = self._filter_respiration_signal()
                    self.respiration_filtered_buffer.append(resp_filtered)
                    
                    # Update respiration rate every 2 seconds
                    current_time = len(self.respiration_filtered_buffer) / self.fps
                    if current_time - self.last_rr_update >= 2.0:
                        self._update_respiration_rate()
                        self.last_rr_update = current_time
    
    def _calculate_respiration_raw(self):
        """Calculate raw respiration signal from shoulder movement"""
        if len(self.shoulder_y_buffer) < 30:
            return 0
        
        # Get last 30 samples (1 second)
        y_window = np.array(list(self.shoulder_y_buffer)[-30:])
        
        # Calculate movement signal (derivative)
        if len(y_window) > 1:
            movement = np.diff(y_window)
            return np.mean(movement)
        
        return 0
    
    def _filter_respiration_signal(self):
        """Apply bandpass filter to respiration signal for breathing frequencies"""
        if len(self.respiration_raw_buffer) < 60:  # At least 2 seconds
            return list(self.respiration_raw_buffer)[-1] if self.respiration_raw_buffer else 0
        
        # Get recent signal
        signal_data = np.array(list(self.respiration_raw_buffer)[-60:])  # Last 2 seconds
        
        # Bandpass filter for respiration rate (0.1-0.8 Hz, 6-48 breaths/min)
        nyquist = self.fps / 2
        low_freq = 0.1 / nyquist  # 6 breaths/min
        high_freq = 0.8 / nyquist  # 48 breaths/min
        
        try:
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            return filtered_signal[-1]
        except:
            return signal_data[-1]
    
    def _update_respiration_rate(self):
        """Estimate respiration rate from filtered signal"""
        if len(self.respiration_filtered_buffer) < 180:  # At least 6 seconds
            return
        
        # Get last 6 seconds of data
        signal_data = np.array(list(self.respiration_filtered_buffer)[-180:])
        
        # Remove DC component
        signal_data = signal_data - np.mean(signal_data)
        
        # Apply window function
        windowed_signal = signal_data * np.hanning(len(signal_data))
        
        # FFT
        fft_signal = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(windowed_signal), 1/self.fps)
        
        # Only consider positive frequencies in respiration range
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_signal[:len(fft_signal)//2])
        
        # Respiration frequency range (0.1-0.8 Hz)
        resp_mask = (positive_freqs >= 0.1) & (positive_freqs <= 0.8)
        resp_freqs = positive_freqs[resp_mask]
        resp_fft = positive_fft[resp_mask]
        
        if len(resp_freqs) > 0:
            # Find peak frequency
            peak_idx = np.argmax(resp_fft)
            peak_freq = resp_freqs[peak_idx]
            
            # Convert to breaths per minute
            self.respiration_rate = peak_freq * 60
    
    def get_shoulder_signal(self):
        """Get current shoulder Y signal"""
        return list(self.shoulder_y_buffer)
    
    def get_respiration_signals(self):
        """Get respiration signals"""
        return {
            'raw': list(self.respiration_raw_buffer),
            'filtered': list(self.respiration_filtered_buffer)
        }
    
    def get_respiration_rate(self):
        """Get current respiration rate estimate"""
        return self.respiration_rate