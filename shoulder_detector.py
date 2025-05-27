import cv2
import numpy as np
import mediapipe as mp

class ShoulderMovementDetector:
    """Detector pergerakan bahu untuk ekstraksi sinyal respirasi"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_shoulder_landmarks(self, frame):
        """Deteksi landmark bahu dari frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get shoulder landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            h, w, _ = frame.shape
            
            # Convert to pixel coordinates
            left_shoulder_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_pos = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            
            return left_shoulder_pos, right_shoulder_pos, results.pose_landmarks
        
        return None, None, None
    
    def calculate_shoulder_movement(self, left_pos, right_pos):
        """Kalkulasi pergerakan bahu untuk respirasi"""
        if left_pos is None or right_pos is None:
            return None
            
        # Calculate vertical movement (breathing causes up-down movement)
        shoulder_center_y = (left_pos[1] + right_pos[1]) / 2
        
        # Calculate shoulder width (breathing can affect shoulder distance)
        shoulder_distance = np.sqrt((left_pos[0] - right_pos[0])**2 + (left_pos[1] - right_pos[1])**2)
        
        return {
            'center_y': shoulder_center_y,
            'distance': shoulder_distance,
            'left_y': left_pos[1],
            'right_y': right_pos[1]
        }
    
    def draw_shoulder_landmarks(self, frame, left_pos, right_pos, pose_landmarks):
        """Gambar landmark dan garis pada frame"""
        if left_pos and right_pos:
            # Draw shoulder points
            cv2.circle(frame, left_pos, 8, (0, 255, 0), -1)
            cv2.circle(frame, right_pos, 8, (0, 255, 0), -1)
            
            # Draw line between shoulders
            cv2.line(frame, left_pos, right_pos, (255, 0, 0), 3)
            
            # Draw center point
            center_x = (left_pos[0] + right_pos[0]) // 2
            center_y = (left_pos[1] + right_pos[1]) // 2
            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
            
            # Add labels
            cv2.putText(frame, "L", (left_pos[0]-15, left_pos[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "R", (right_pos[0]+10, right_pos[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "C", (center_x+10, center_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw full pose if available
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return frame