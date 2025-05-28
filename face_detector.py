import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_indices = [
            10,  103, 14, 332
        ]
    
    def detect_face_roi(self, frame):
        """
        Detect face and return ROI coordinates and landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Get ROI points
            roi_points = []
            for idx in self.face_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                roi_points.append([x, y])
            
            roi_points = np.array(roi_points)
            
            # Create bounding box
            x_min, y_min = np.min(roi_points, axis=0)
            x_max, y_max = np.max(roi_points, axis=0)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Create mask for the face region
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], 255)
            
            return {
                'bbox': (x_min, y_min, x_max, y_max),
                'mask': mask,
                'landmarks': face_landmarks,
                'roi_points': roi_points
            }
        
        return None
    
    def extract_rgb_signals(self, frame, face_data):
        """
        Extract RGB signals from face region
        """
        if face_data is None:
            return None, None, None
        
        x_min, y_min, x_max, y_max = face_data['bbox']
        mask = face_data['mask']
        
        # Extract face region
        face_region = frame[y_min:y_max, x_min:x_max]
        mask_region = mask[y_min:y_max, x_min:x_max]
        
        if face_region.size == 0 or mask_region.size == 0:
            return None, None, None
        
        # Apply mask and calculate mean RGB values
        face_region_masked = cv2.bitwise_and(face_region, face_region, mask=mask_region)
        
        # Calculate mean values for each channel
        b_mean = np.mean(face_region_masked[:, :, 0][mask_region > 0])
        g_mean = np.mean(face_region_masked[:, :, 1][mask_region > 0])
        r_mean = np.mean(face_region_masked[:, :, 2][mask_region > 0])
        
        return r_mean, g_mean, b_mean