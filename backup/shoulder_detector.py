import cv2
import numpy as np
import mediapipe as mp

class ShoulderDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Shoulder landmark indices
        self.left_shoulder_idx = 11
        self.right_shoulder_idx = 12
        
    def detect_shoulders(self, frame):
        """
        Detect shoulder landmarks and return their positions
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Get shoulder positions
            left_shoulder = landmarks[self.left_shoulder_idx]
            right_shoulder = landmarks[self.right_shoulder_idx]
            
            # Convert normalized coordinates to pixel coordinates
            left_x = int(left_shoulder.x * w)
            left_y = int(left_shoulder.y * h)
            right_x = int(right_shoulder.x * w)
            right_y = int(right_shoulder.y * h)
            
            # Calculate midpoint
            mid_x = (left_x + right_x) // 2
            mid_y = (left_y + right_y) // 2
            
            return {
                'left_shoulder': (left_x, left_y),
                'right_shoulder': (right_x, right_y),
                'midpoint': (mid_x, mid_y),
                'landmarks': results.pose_landmarks
            }
        
        return None
    
    def draw_landmarks(self, frame, shoulder_data):
        """
        Draw shoulder landmarks on frame
        """
        if shoulder_data is not None:
            left_x, left_y = shoulder_data['left_shoulder']
            right_x, right_y = shoulder_data['right_shoulder']
            mid_x, mid_y = shoulder_data['midpoint']
            
            # Draw shoulder points
            cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (mid_x, mid_y), 8, (255, 0, 0), -1)
            
            # Draw line between shoulders
            cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2)
            
            # Add text
            cv2.putText(frame, f"Shoulder Y: {mid_y}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame