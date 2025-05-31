import numpy as np
import time
from collections import deque
from scipy import signal
from scipy.signal import savgol_filter

class SignalProcessor:
    def __init__(self, fps=20, buffer_size=200):
        self.fps = fps
        self.buffer_size = buffer_size
        self.min_window = int(1.6 * fps)  # 1.6 seconds for POS
        
        # Filtering parameters
        self.heart_rate_low_freq = 0.8
        self.heart_rate_high_freq = 2.8
        self.respiration_low_freq = 0.1
        self.respiration_high_freq = 0.8
        
        # Moving window smoothing parameters
        self.shoulder_smoothing_window = 15  # Window size for shoulder signal smoothing
        
        # Data buffers
        self.rgb_buffer = deque(maxlen=buffer_size)
        self.pos_raw_buffer = deque(maxlen=buffer_size)
        self.pos_filtered_buffer = deque(maxlen=buffer_size)
        self.pos_savgol_buffer = deque(maxlen=buffer_size)
        self.shoulder_buffer = deque(maxlen=buffer_size)
        self.resp_raw_buffer = deque(maxlen=buffer_size)
        self.resp_filtered_buffer = deque(maxlen=buffer_size)
        
        # Manual detection
        self.heartbeat_taps = deque(maxlen=20)
        self.respiration_taps = deque(maxlen=10)
        
        # Current values
        self.heart_rate = 0
        self.respiration_rate = 0
        self.manual_heart_rate = 0
        self.manual_respiration_rate = 0
        
    
    def simple_pos_algorithm(self, signal):
        """POS method on CPU using Numpy."""
        eps = 1e-9
        X = signal
        
        # Ensure we have the right shape [estimators, channels, frames]
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        
        e, c, f = X.shape
        
        # Check minimum window size
        if f < self.min_window:
            return np.zeros(f)
        
        P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
        
        # Initialize output
        heart_rate_signal = np.zeros((e, f), dtype=np.float64)
        
        for n in range(self.min_window-1, f):
            # Start index of sliding window 
            m = n - self.min_window + 1
            
            # Extract window
            Cn = X[:, :, m:n+1]
            
            # Temporal normalization
            mean_vals = np.mean(Cn, axis=2, keepdims=True)
            mean_vals = np.where(mean_vals == 0, eps, mean_vals)
            Cn_normalized = Cn / mean_vals
            
            # Project to POS space
            for i in range(e):
                # Get the RGB values for this estimator
                rgb_window = Cn_normalized[i, :, :]  # Shape: [3, window_length]
                
                # Apply POS projection
                S = np.dot(P, rgb_window)  # Shape: [2, window_length]
                
                # Tuning step
                S1 = S[0, :]
                S2 = S[1, :]
                
                std1 = np.std(S1)
                std2 = np.std(S2)
                
                if std2 > eps:
                    alpha = std1 / std2
                else:
                    alpha = 1.0
                
                # Combine signals
                Hn = S1 + alpha * S2
                
                # Remove mean
                Hn = Hn - np.mean(Hn)
                
                # Overlap-add
                heart_rate_signal[i, m:n+1] += Hn
        
        return heart_rate_signal.flatten()
    
    def bandpass_filter(self, data, low_freq, high_freq):
        """Simple bandpass filter"""
        if len(data) < 60:  # Reduced minimum length
            return data[-1] if data else 0
        
        data_array = np.array(list(data)[-60:])
        nyquist = self.fps / 2
        low, high = low_freq / nyquist, high_freq / nyquist
        
        # Ensure frequencies are within valid range
        low = max(low, 0.01)
        high = min(high, 0.99)
        
        print(f"Applying bandpass filter: low={low}, high={high}, data length={len(data_array)}")
        
        try:
            b, a = signal.butter(3, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data_array)
            return filtered[-1]
        except:
            return data_array[-1]
    
    def apply_savgol_filter(self, data, window_length=11, polyorder=3):
        """Apply Savitzky-Golay filter"""
        if len(data) < window_length:
            return data[-1] if data else 0
        
        data_array = np.array(list(data)[-window_length:])
        try:
            filtered = savgol_filter(data_array, window_length, polyorder)
            return filtered[-1]
        except:
            return data_array[-1]
    
    def estimate_heart_rate(self):
        """Estimate heart rate using FFT"""
        if len(self.pos_savgol_buffer) < 150:
            return
        
        data = np.array(list(self.pos_savgol_buffer)[-150:])
        data = data - np.mean(data)
        
        windowed = data * np.hanning(len(data))
        fft_vals = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(len(windowed), 1/self.fps)
        
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        hr_mask = (pos_freqs >= self.heart_rate_low_freq) & (pos_freqs <= self.heart_rate_high_freq)
        if np.any(hr_mask):
            hr_freqs = pos_freqs[hr_mask]
            hr_power = pos_fft[hr_mask]
            peak_freq = hr_freqs[np.argmax(hr_power)]
            self.heart_rate = peak_freq * 60
    
    def estimate_respiration_rate(self):
        """Estimate respiration rate using FFT"""
        if len(self.resp_filtered_buffer) < 180:
            return
        
        data = np.array(list(self.resp_filtered_buffer)[-180:])
        data = data - np.mean(data)
        
        windowed = data * np.hanning(len(data))
        fft_vals = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(len(windowed), 1/self.fps)
        
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        resp_mask = (pos_freqs >= self.respiration_low_freq) & (pos_freqs <= self.respiration_high_freq)
        if np.any(resp_mask):
            resp_freqs = pos_freqs[resp_mask]
            resp_power = pos_fft[resp_mask]
            peak_freq = resp_freqs[np.argmax(resp_power)]
            self.respiration_rate = peak_freq * 60
    
    def process_rgb_signal(self, r, g, b):
        """Process RGB signal for heart rate"""
        if r is not None and not np.isnan(r) and not np.isnan(g) and not np.isnan(b):
            self.rgb_buffer.append([r, g, b])
            
            # POS processing
            if len(self.rgb_buffer) >= self.min_window:
                # Get recent RGB data and reshape for POS
                recent_rgb = np.array(list(self.rgb_buffer)[-self.min_window:])
                recent_rgb = recent_rgb.T  # Shape: [3, frames]
                recent_rgb = recent_rgb.reshape(1, 3, -1)  # Shape: [1, 3, frames]
                
                # Apply POS algorithm
                pos_signal = self.simple_pos_algorithm(recent_rgb)
                
                # Add only the latest value to avoid accumulation
                if len(pos_signal) > 0:
                    self.pos_raw_buffer.append(pos_signal[-1])
                    
                    # Bandpass filter
                    if len(self.pos_raw_buffer) >= 30:
                        pos_filtered = self.bandpass_filter(self.pos_raw_buffer, self.heart_rate_low_freq, self.heart_rate_high_freq)
                        self.pos_filtered_buffer.append(pos_filtered)
                        
                        # Savitzky-Golay filter
                        if len(self.pos_filtered_buffer) >= 11:
                            pos_savgol = self.apply_savgol_filter(self.pos_filtered_buffer, 14, 4)
                            self.pos_savgol_buffer.append(pos_savgol)
                            
                            # Update heart rate every second
                            if len(self.pos_savgol_buffer) % self.fps == 0:
                                self.estimate_heart_rate()
    
    def process_shoulder_signal(self, shoulder_y):
        """Process shoulder movement for respiration with moving window smoothing"""
        if shoulder_y is not None:
            self.shoulder_buffer.append(shoulder_y)
            
            # Calculate raw respiration signal from smoothed shoulder movement
            if len(self.shoulder_buffer) >= 30:
                    # Use smoothed values for movement calculation
                    y_window = np.array(list(self.shoulder_buffer)[-30:])
                    
                    # Apply moving window smoothing to the entire window for better signal quality
                    smoothed_window = []
                    for i in range(len(y_window)):
                        start_idx = max(0, i - self.shoulder_smoothing_window // 2)
                        end_idx = min(len(y_window), i + self.shoulder_smoothing_window // 2 + 1)
                        smoothed_window.append(np.mean(y_window[start_idx:end_idx]))
                    
                    smoothed_window = np.array(smoothed_window)
                    
                    if len(smoothed_window) > 1:
                        # Calculate movement using gradient on smoothed signal
                        movement = np.mean(np.gradient(smoothed_window))
                        self.resp_raw_buffer.append(movement)
                        
                        # Apply bandpass filter
                        if len(self.resp_raw_buffer) >= 30:
                            resp_filtered = self.bandpass_filter(self.resp_raw_buffer, self.respiration_low_freq, self.respiration_high_freq)
                            self.resp_filtered_buffer.append(resp_filtered)
                            
                            # Update respiration rate every 2 seconds
                            if len(self.resp_filtered_buffer) % (self.fps * 2) == 0:
                                self.estimate_respiration_rate()
    
    def tap_heartbeat(self):
        """Manual heartbeat tap"""
        current_time = time.time()
        self.heartbeat_taps.append(current_time)
        
        if len(self.heartbeat_taps) >= 2:
            recent_taps = list(self.heartbeat_taps)[-10:]
            intervals = []
            for i in range(1, len(recent_taps)):
                interval = recent_taps[i] - recent_taps[i-1]
                if 0.3 <= interval <= 2.0:
                    intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                self.manual_heart_rate = 60.0 / avg_interval
    
    def tap_respiration(self):
        """Manual respiration tap"""
        current_time = time.time()
        self.respiration_taps.append(current_time)
        
        if len(self.respiration_taps) >= 2:
            recent_taps = list(self.respiration_taps)[-5:]
            intervals = []
            for i in range(1, len(recent_taps)):
                interval = recent_taps[i] - recent_taps[i-1]
                if 1.0 <= interval <= 10.0:
                    intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                self.manual_respiration_rate = 60.0 / avg_interval
    
    def reset_heartbeat(self):
        """Reset heartbeat"""
        self.heartbeat_taps.clear()
        self.manual_heart_rate = 0
        
    def reset_respiration(self):
        """Reset respiration"""
        self.respiration_taps.clear()
        self.manual_respiration_rate = 0