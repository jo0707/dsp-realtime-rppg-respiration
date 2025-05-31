import cv2
import numpy as np
import mediapipe as mp
import time

class CameraProcessor:
    def __init__(self, camera_index=0, fps=30):
        self.fps = fps
        self.camera_index = camera_index

        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        self.fps_update_interval = fps  # Update FPS every fps frames

        # Lucas-Kanade tracking variables
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Face tracking variables
        self.face_points = None
        self.prev_gray = None
        self.face_tracked = False

        # Shoulder tracking variables
        self.shoulder_points = None
        self.shoulder_prev_gray = None
        self.track_shoulders = False
        self.shoulder_detection_interval = 100  # Re-detect shoulders every 30 frames
        self.frames_since_shoulder_detection = 0

        # MediaPipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (2 meters), 1 for full-range (5 meters)
            min_detection_confidence=0.5
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % self.fps_update_interval == 0:
            elapsed_time = time.time() - self.start_time
            self.current_fps = self.fps_update_interval / elapsed_time
            self.start_time = time.time()

    def detect_face_initial(self, frame):
        """Initial face detection using MediaPipe Face Detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]  # Use the first detected face
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Define forehead region (upper 30% of face)
            forehead_y = y + int(height * 0)  # Start 10% up from top
            forehead_height = int(height * 0.9)  # Take 30% of face height
            forehead_x = x + int(width * 0.1)   # Start 20% from left
            forehead_width = int(width * 0.7)   # Take 60% of face width

            # Create grid of tracking points in forehead region
            points_per_row = 4
            points_per_col = 3
            key_points = []

            for i in range(points_per_col):
                for j in range(points_per_row):
                    px = forehead_x + (j * forehead_width // (points_per_row - 1))
                    py = forehead_y + (i * forehead_height // (points_per_col - 1))

                    # Ensure points are within frame bounds
                    if 0 <= px < w and 0 <= py < h:
                        key_points.append([px, py])

            if len(key_points) >= 8:  # Need minimum points for tracking
                self.face_points = np.array(key_points, dtype=np.float32).reshape(-1, 1, 2)
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.face_tracked = True
                return True

        return False

    def track_face_lk(self, frame):
        """Track face using Lucas-Kanade optical flow"""
        if self.face_points is None or self.prev_gray is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.face_points, None, **self.lk_params)

            # Check if we got valid results
            if new_points is None or status is None:
                return None

            # Filter good points - fix the indexing issue
            status = status.flatten()  # Ensure status is 1D
            good_new = new_points[status == 1]

            # Check if we have enough good points
            if len(good_new) < 6:
                return None

            # Update points and frame
            self.face_points = good_new.reshape(-1, 1, 2)
            self.prev_gray = gray.copy()

            # Calculate bounding box from tracked points
            x_coords = self.face_points[:, 0, 0]
            y_coords = self.face_points[:, 0, 1]

            x_min = max(int(np.min(x_coords)) - 20, 0)
            y_min = max(int(np.min(y_coords)) - 20, 0)
            x_max = min(int(np.max(x_coords)) + 20, frame.shape[1])
            y_max = min(int(np.max(y_coords)) + 20, frame.shape[0])

            return (x_min, y_min, x_max, y_max)

        except Exception as e:
            # If tracking fails, return None to trigger re-detection
            return None

    def detect_face(self, frame):
        """Improved face detection with Lucas-Kanade tracking"""

        # Re-detect if tracking failed
        if not self.face_tracked:
            self.face_tracked = self.detect_face_initial(frame)
            if not self.face_tracked:
                return None, None, None
        else:
            roi_coords = self.track_face_lk(frame)
            if roi_coords is None:
                self.face_tracked = False
                return None, None, None

            x_min, y_min, x_max, y_max = roi_coords

            # Extract forehead ROI for rPPG signal
            face_region = frame[y_min:y_max, x_min:x_max]

            if face_region.size > 0:
                # Calculate mean RGB values
                b_mean = np.mean(face_region[:, :, 0])
                g_mean = np.mean(face_region[:, :, 1])
                r_mean = np.mean(face_region[:, :, 2])

                # Draw face ROI and tracking points
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw tracking points
                for point in self.face_points:
                    x, y = point.ravel()
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                return r_mean, g_mean, b_mean

        return None, None, None

    def detect_shoulders_initial(self, frame):
        """Initial shoulder detection using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Get shoulder points for tracking
            left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)

            # Ensure points are within frame bounds
            if (0 <= left_x < w and 0 <= left_y < h and
                0 <= right_x < w and 0 <= right_y < h):

                # Store both shoulder points for tracking
                key_points = [[left_x, left_y], [right_x, right_y]]
                self.shoulder_points = np.array(key_points, dtype=np.float32).reshape(-1, 1, 2)
                self.shoulder_prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.track_shoulders = True
                return True

        return False

    def track_shoulders_lk(self, frame):
        """Track shoulders using Lucas-Kanade optical flow"""
        if self.shoulder_points is None or self.shoulder_prev_gray is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.shoulder_prev_gray, gray, self.shoulder_points, None, **self.lk_params)

            # Check if we got valid results
            if new_points is None or status is None:
                return None

            # Filter good points
            status = status.flatten()
            good_new = new_points[status == 1]

            # Check if we have both shoulder points
            if len(good_new) < 2:
                return None

            # Update points and frame
            self.shoulder_points = good_new.reshape(-1, 1, 2)
            self.shoulder_prev_gray = gray.copy()

            # Extract shoulder coordinates
            left_shoulder = self.shoulder_points[0, 0]
            right_shoulder = self.shoulder_points[1, 0]

            left_x, left_y = int(left_shoulder[0]), int(left_shoulder[1])
            right_x, right_y = int(right_shoulder[0]), int(right_shoulder[1])

            return (left_x, left_y, right_x, right_y)

        except Exception as e:
            # If tracking fails, return None to trigger re-detection
            return None

    def detect_shoulders(self, frame):
        """Improved shoulder detection with Lucas-Kanade tracking"""
        self.frames_since_shoulder_detection += 1

        # Re-detect shoulders periodically or if tracking failed
        if not self.track_shoulders or self.frames_since_shoulder_detection >= self.shoulder_detection_interval:
            if self.detect_shoulders_initial(frame):
                self.frames_since_shoulder_detection = 0
            else:
                self.track_shoulders = False
                return None

        # Use Lucas-Kanade tracking
        if self.track_shoulders:
            shoulder_coords = self.track_shoulders_lk(frame)
            if shoulder_coords is None:
                # Tracking failed, force re-detection
                self.track_shoulders = False
                self.frames_since_shoulder_detection = self.shoulder_detection_interval
                return None

            left_x, left_y, right_x, right_y = shoulder_coords
            mid_y = (left_y + right_y) // 2

            cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2)

            # Draw tracking points
            for point in self.shoulder_points:
                x, y = point.ravel()
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 255), -1)

            return mid_y

        return None

    def process_frame(self, frame, signal_processor):
        """Process single frame and extract vital signs"""
        # Update FPS calculation
        self.update_fps()

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Face detection and RGB extraction
        r, g, b = self.detect_face(frame)
        signal_processor.process_rgb_signal(r, g, b)

        # Shoulder detection
        shoulder_y = self.detect_shoulders(frame)
        signal_processor.process_shoulder_signal(shoulder_y)

        return frame