import time
from collections import deque
import numpy as np

class ManualDetector:
    def __init__(self):
        # Store timestamps of manual taps
        self.heartbeat_taps = deque(maxlen=20)  # Store last 20 taps
        self.respiration_taps = deque(maxlen=10)  # Store last 10 taps
        
        # Calculated rates
        self.manual_heart_rate = 0
        self.manual_respiration_rate = 0
        
    def tap_heartbeat(self):
        """Record a manual heartbeat tap"""
        current_time = time.time()
        self.heartbeat_taps.append(current_time)
        self._calculate_heart_rate()
        
    def tap_respiration(self):
        """Record a manual respiration tap"""
        current_time = time.time()
        self.respiration_taps.append(current_time)
        self._calculate_respiration_rate()
        
    def _calculate_heart_rate(self):
        """Calculate heart rate from tap intervals"""
        if len(self.heartbeat_taps) < 2:
            return
            
        # Use last 10 taps for calculation
        recent_taps = list(self.heartbeat_taps)[-10:]
        if len(recent_taps) < 2:
            return
            
        # Calculate intervals between taps
        intervals = []
        for i in range(1, len(recent_taps)):
            interval = recent_taps[i] - recent_taps[i-1]
            if 0.3 <= interval <= 2.0:  # Valid heartbeat interval (30-200 BPM)
                intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            self.manual_heart_rate = 60.0 / avg_interval  # Convert to BPM
            
    def _calculate_respiration_rate(self):
        """Calculate respiration rate from tap intervals"""
        if len(self.respiration_taps) < 2:
            return
            
        # Use last 5 taps for calculation
        recent_taps = list(self.respiration_taps)[-5:]
        if len(recent_taps) < 2:
            return
            
        # Calculate intervals between taps
        intervals = []
        for i in range(1, len(recent_taps)):
            interval = recent_taps[i] - recent_taps[i-1]
            if 1.0 <= interval <= 10.0:  # Valid breathing interval (6-60 BPM)
                intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            self.manual_respiration_rate = 60.0 / avg_interval  # Convert to BPM
            
    def get_manual_heart_rate(self):
        """Get current manual heart rate"""
        return self.manual_heart_rate
        
    def get_manual_respiration_rate(self):
        """Get current manual respiration rate"""
        return self.manual_respiration_rate
        
    def reset_heartbeat(self):
        """Reset heartbeat taps"""
        self.heartbeat_taps.clear()
        self.manual_heart_rate = 0
        
    def reset_respiration(self):
        """Reset respiration taps"""
        self.respiration_taps.clear()
        self.manual_respiration_rate = 0