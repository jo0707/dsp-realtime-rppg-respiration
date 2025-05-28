import numpy as np
from scipy import signal
from collections import deque

def pos_algorithm(rgb_signals, fps):
    """
    Simplified POS algorithm for heart rate detection.
    
    Args:
        rgb_signals: numpy array of shape (3, N) where rows are [R, G, B] and N is number of frames
        fps: frame rate
    
    Returns:
        pos_signal: 1D array of POS signal values
    """
    if rgb_signals.shape[1] < int(1.6 * fps):
        return np.array([])
    
    # Normalize RGB signals
    rgb_norm = rgb_signals / (np.mean(rgb_signals, axis=1, keepdims=True) + 1e-9)
    
    # POS projection matrix
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    
    # Apply projection
    projected = np.dot(P, rgb_norm)
    S1, S2 = projected[0], projected[1]
    
    # Calculate alpha (tuning parameter)
    alpha = np.std(S1) / (np.std(S2) + 1e-9)
    
    # Generate POS signal
    pos_signal = S1 - alpha * S2
    
    return pos_signal

class POSProcessor:
    def __init__(self, fps=30, buffer_seconds=10):
        self.fps = fps
        self.buffer_size = fps * buffer_seconds
        self.min_window = int(1.6 * fps)  # Minimum 1.6 seconds for POS
        
        # RGB signal buffers
        self.rgb_buffer = deque(maxlen=self.buffer_size)
        
        # POS signal buffers  
        self.pos_raw = deque(maxlen=self.buffer_size)
        self.pos_filtered = deque(maxlen=self.buffer_size)
        
        # Heart rate
        self.heart_rate = 0
        self.last_update = 0
        
    def add_rgb_sample(self, r, g, b):
        """Add new RGB sample and process POS signal"""
        if None in [r, g, b]:
            return
            
        self.rgb_buffer.append([r, g, b])
        
        # Process when we have enough data
        if len(self.rgb_buffer) >= self.min_window:
            # Get recent RGB data
            recent_rgb = np.array(list(self.rgb_buffer)[-self.min_window:]).T
            
            # Calculate POS signal
            pos_signal = pos_algorithm(recent_rgb, self.fps)
            
            if len(pos_signal) > 0:
                # Store raw POS value (latest)
                self.pos_raw.append(pos_signal[-1])
                
                # Apply bandpass filter if we have enough data
                if len(self.pos_raw) >= 90:  # 3 seconds
                    filtered_value = self._bandpass_filter()
                    self.pos_filtered.append(filtered_value)
                    
                    # Update heart rate every second
                    if len(self.pos_filtered) - self.last_update >= self.fps:
                        self._estimate_heart_rate()
                        self.last_update = len(self.pos_filtered)
    
    def _bandpass_filter(self):
        """Apply bandpass filter for heart rate frequencies (0.7-4 Hz)"""
        if len(self.pos_raw) < 90:
            return list(self.pos_raw)[-1]
        
        # Get last 3 seconds of data
        data = np.array(list(self.pos_raw)[-90:])
        
        # Bandpass filter (42-240 BPM)
        nyquist = self.fps / 2
        low, high = 0.7 / nyquist, 4.0 / nyquist
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            return filtered[-1]
        except:
            return data[-1]
    
    def _estimate_heart_rate(self):
        """Estimate heart rate using FFT on filtered signal"""
        if len(self.pos_filtered) < 150:  # Need 5 seconds
            return
        
        # Get 5 seconds of filtered data
        data = np.array(list(self.pos_filtered)[-150:])
        data = data - np.mean(data)  # Remove DC
        
        # Apply window and FFT
        windowed = data * np.hanning(len(data))
        fft_vals = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(len(windowed), 1/self.fps)
        
        # Find peak in heart rate range (0.7-4 Hz)
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        hr_mask = (pos_freqs >= 0.7) & (pos_freqs <= 4.0)
        if np.any(hr_mask):
            hr_freqs = pos_freqs[hr_mask]
            hr_power = pos_fft[hr_mask]
            peak_freq = hr_freqs[np.argmax(hr_power)]
            self.heart_rate = peak_freq * 60  # Convert to BPM
    
    def get_rgb_signals(self):
        """Get RGB signals as separate lists"""
        if not self.rgb_buffer:
            return {'red': [], 'green': [], 'blue': []}
        
        rgb_array = np.array(self.rgb_buffer)
        return {
            'red': rgb_array[:, 0].tolist(),
            'green': rgb_array[:, 1].tolist(), 
            'blue': rgb_array[:, 2].tolist()
        }
    
    def get_pos_signals(self):
        """Get POS signals"""
        return {
            'raw': list(self.pos_raw),
            'filtered': list(self.pos_filtered)
        }
    
    def get_heart_rate(self):
        """Get current heart rate estimate"""
        return self.heart_rate