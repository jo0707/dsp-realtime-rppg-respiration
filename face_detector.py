import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    """Detector wajah sederhana menggunakan MediaPipe untuk ekstraksi ROI"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
    def detect_face_roi(self, frame):
        """Ekstrak region of interest dari wajah"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ekstrak area dahi (30% bagian atas wajah)
            forehead_height = int(height * 0.3)
            roi = frame[y:y+forehead_height, x:x+width]
            
            return roi, (x, y, width, forehead_height), detection.score[0]
        
        return None, None, 0