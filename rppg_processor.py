import numpy as np
from scipy import signal
from pos import POS

class rPPGProcessor:
    """Processor rPPG untuk heart rate detection menggunakan metode POS"""
    
    def __init__(self, fps=30, window_size=150):
        self.fps = fps
        self.window_size = window_size
        self.rgb_buffer = []
        
    def extract_rgb_signals(self, roi):
        """Ekstrak nilai RGB rata-rata dari ROI"""
        if roi is None or roi.size == 0:
            return None
            
        r_mean = np.mean(roi[:, :, 2])
        g_mean = np.mean(roi[:, :, 1])
        b_mean = np.mean(roi[:, :, 0])
        
        return [r_mean, g_mean, b_mean]
    
    def bandpass_filter(self, signal_data, low_freq=0.7, high_freq=4.0):
        """Bandpass filter untuk heart rate range (42-240 BPM)"""
        if len(signal_data) < 10:
            return signal_data
            
        nyquist = self.fps / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def process_frame(self, roi):
        """Proses frame dan return sinyal heart rate"""
        rgb_values = self.extract_rgb_signals(roi)
        
        if rgb_values is None:
            return None, None, None
            
        self.rgb_buffer.append(rgb_values)
        
        if len(self.rgb_buffer) > self.window_size:
            self.rgb_buffer.pop(0)
            
        if len(self.rgb_buffer) < 50:
            return rgb_values, None, None
            
        # Convert to POS format: (1, 3, frames)
        rgb_array = np.array(self.rgb_buffer).T
        pos_input = np.expand_dims(rgb_array, axis=0)
        
        # Apply POS method
        pos_signal = POS(pos_input, self.fps)
        raw_signal = pos_signal[0]
        
        # Filter for heart rate
        processed_signal = self.bandpass_filter(raw_signal)
        
        return rgb_values, raw_signal, processed_signal