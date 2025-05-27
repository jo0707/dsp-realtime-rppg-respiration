import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

class HeartRateProcessor:
    """Kalkulasi heart rate dari sinyal rPPG"""
    
    def __init__(self, fps=30):
        self.fps = fps
        
    def calculate_heart_rate(self, processed_signal):
        """Kalkulasi heart rate menggunakan FFT"""
        if processed_signal is None or len(processed_signal) < 30:
            return 0
            
        windowed_signal = processed_signal * signal.windows.hann(len(processed_signal))
        
        fft_values = fft(windowed_signal)
        freqs = fftfreq(len(windowed_signal), 1/self.fps)
        
        # Heart rate range (40-180 BPM)
        hr_range = (freqs >= 0.67) & (freqs <= 3.0)
        
        if not np.any(hr_range):
            return 0
            
        power_spectrum = np.abs(fft_values[hr_range])
        dominant_freq_idx = np.argmax(power_spectrum)
        dominant_freq = freqs[hr_range][dominant_freq_idx]
        
        heart_rate = dominant_freq * 60
        
        return heart_rate