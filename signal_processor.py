import numpy as np
import time
from collections import deque
from scipy import signal
from scipy.signal import savgol_filter

class SignalProcessor:
    def __init__(self, fps=30, buffer_size=300):
        self.fps = fps
        self.buffer_size = buffer_size
        self.min_window = int(1.6 * fps)  # 1.6 seconds for POS
        
        # Data buffers
        self.rgb_buffer = deque(maxlen=buffer_size)
        self.pos_raw_buffer = deque(maxlen=buffer_size)
        self.pos_filtered_buffer = deque(maxlen=buffer_size)
        self.pos_savgol_buffer = deque(maxlen=buffer_size)
        self.shoulder_buffer = deque(maxlen=buffer_size)
        self.shoulder_smoothed_buffer = deque(maxlen=buffer_size)  # New smoothed shoulder buffer
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
        
        # Shoulder smoothing parameters
        self.shoulder_moving_avg_window = 5
        self.outlier_threshold = 2.0  # Standard deviations for outlier detection
    
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
    
    def apply_moving_average(self, data, window_size=5):
        """Apply moving average filter for smoothing"""
        if len(data) < window_size:
            return data[-1] if data else 0
        
        data_array = np.array(list(data)[-window_size:])
        return np.mean(data_array)
    
    def remove_outliers(self, value, buffer, threshold=2.0):
        """Remove outliers using z-score method"""
        if len(buffer) < 10:
            return value
        
        data_array = np.array(list(buffer)[-20:])  # Use last 20 values for statistics
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        
        if std_val == 0:
            return value
        
        z_score = abs(value - mean_val) / std_val
        
        # If outlier detected, use median of recent values
        if z_score > threshold:
            return np.median(data_array[-5:])
        
        return value
    
    def smooth_shoulder_signal(self, shoulder_y):
        """Enhanced shoulder signal smoothing with multiple filtering stages"""
        if shoulder_y is None:
            return None
        
        # Stage 1: Outlier removal
        cleaned_value = self.remove_outliers(shoulder_y, self.shoulder_buffer, self.outlier_threshold)
        
        # Stage 2: Add to buffer for further processing
        temp_buffer = list(self.shoulder_buffer) + [cleaned_value]
        
        # Stage 3: Moving average smoothing
        if len(temp_buffer) >= self.shoulder_moving_avg_window:
            smoothed_value = self.apply_moving_average(temp_buffer, self.shoulder_moving_avg_window)
        else:
            smoothed_value = cleaned_value
        
        # Stage 4: Additional Savitzky-Golay smoothing if we have enough data
        temp_smoothed_buffer = list(self.shoulder_smoothed_buffer) + [smoothed_value]
        if len(temp_smoothed_buffer) >= 11:
            final_smoothed = self.apply_savgol_filter(temp_smoothed_buffer, 11, 3)
        else:
            final_smoothed = smoothed_value
        
        return final_smoothed
    
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
        
        hr_mask = (pos_freqs >= 0.7) & (pos_freqs <= 4.0)
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
        
        resp_mask = (pos_freqs >= 0.1) & (pos_freqs <= 0.8)
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
                        pos_filtered = self.bandpass_filter(self.pos_raw_buffer, 0.8, 2.8)
                        self.pos_filtered_buffer.append(pos_filtered)
                        
                        # Savitzky-Golay filter
                        if len(self.pos_filtered_buffer) >= 11:
                            pos_savgol = self.apply_savgol_filter(self.pos_filtered_buffer, 14, 4)
                            self.pos_savgol_buffer.append(pos_savgol)
                            
                            # Update heart rate every second
                            if len(self.pos_savgol_buffer) % self.fps == 0:
                                self.estimate_heart_rate()
    
    def process_shoulder_signal(self, shoulder_y):
        """Process shoulder movement for respiration with enhanced smoothing"""
        if shoulder_y is not None:
            # Apply enhanced smoothing to the shoulder signal
            smoothed_shoulder_y = self.smooth_shoulder_signal(shoulder_y)
            
            # Add both raw and smoothed values to their respective buffers
            self.shoulder_buffer.append(shoulder_y)
            if smoothed_shoulder_y is not None:
                self.shoulder_smoothed_buffer.append(smoothed_shoulder_y)
            
            # Use smoothed buffer for respiration processing if available
            buffer_to_use = self.shoulder_smoothed_buffer if len(self.shoulder_smoothed_buffer) >= 30 else self.shoulder_buffer
            
            # Respiration processing with smoothed signals
            if len(buffer_to_use) >= 30:
                y_window = np.array(list(buffer_to_use)[-30:])
                if len(y_window) > 1:
                    # Calculate movement using a more robust method
                    # Use gradient for smoother derivative calculation
                    movement_gradient = np.gradient(y_window)
                    movement = np.mean(movement_gradient)
                    
                    # Additional smoothing on the movement signal itself
                    self.resp_raw_buffer.append(movement)
                    
                    # Enhanced respiration filtering
                    if len(self.resp_raw_buffer) >= 60:
                        # Apply bandpass filter first
                        resp_filtered = self.bandpass_filter(self.resp_raw_buffer, 0.08, 0.7)  # Slightly wider frequency range
                        
                        # Apply Savitzky-Golay filter with optimized parameters for respiration
                        resp_filtered = self.apply_savgol_filter(self.resp_raw_buffer, 15, 3)  # Adjusted window and polynomial order
                        
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