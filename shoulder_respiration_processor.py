import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque

class ShoulderRespirationProcessor:
    """Processor sinyal respirasi dari pergerakan bahu"""
    
    def __init__(self, fps=30, window_size=300):
        self.fps = fps
        self.window_size = window_size
        self.movement_buffer = deque(maxlen=window_size)
        
    def extract_respiration_signal(self, shoulder_data):
        """Ekstrak sinyal respirasi dari data pergerakan bahu"""
        if shoulder_data is None:
            return None
            
        # Use vertical center movement as primary respiration signal
        respiration_signal = shoulder_data['center_y']
        
        return respiration_signal
    
    def bandpass_filter(self, signal_data, low_freq=0.1, high_freq=0.8):
        """Filter untuk range frekuensi respirasi (6-48 BPM)"""
        if len(signal_data) < 20:
            return signal_data
            
        nyquist = self.fps / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
            return filtered
        except:
            return signal_data
    
    def calculate_respiration_rate(self, processed_signal):
        """Kalkulasi breathing rate dari sinyal yang difilter"""
        if processed_signal is None or len(processed_signal) < 60:
            return 0
            
        windowed_signal = processed_signal * signal.windows.hann(len(processed_signal))
        
        fft_values = fft(windowed_signal)
        freqs = fftfreq(len(windowed_signal), 1/self.fps)
        
        # Respiration range (8-40 BPM)
        resp_range = (freqs >= 0.13) & (freqs <= 0.67)
        
        if not np.any(resp_range):
            return 0
            
        power_spectrum = np.abs(fft_values[resp_range])
        dominant_freq_idx = np.argmax(power_spectrum)
        dominant_freq = freqs[resp_range][dominant_freq_idx]
        
        respiration_rate = dominant_freq * 60
        
        return respiration_rate
    
    def process_movement(self, shoulder_data):
        """Proses data pergerakan bahu untuk mendapatkan sinyal respirasi"""
        raw_signal = self.extract_respiration_signal(shoulder_data)
        
        if raw_signal is None:
            return None, None, 0
            
        self.movement_buffer.append(raw_signal)
        
        if len(self.movement_buffer) < 60:
            return raw_signal, None, 0
            
        # Convert to numpy array and normalize
        signal_array = np.array(list(self.movement_buffer))
        signal_array = signal_array - np.mean(signal_array)
        
        # Filter for respiration frequency
        filtered_signal = self.bandpass_filter(signal_array)
        
        # Calculate respiration rate
        respiration_rate = self.calculate_respiration_rate(filtered_signal)
        
        return raw_signal, filtered_signal, respiration_rate