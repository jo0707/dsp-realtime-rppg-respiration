import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Fungsi untuk mendeteksi wajah dari webcam
def detect_faces_from_webcam():
    # Membuka webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam tidak tersedia.")
        return

    # Inisialisasi Deteksi Wajah
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Loop frame-by-frame dari webcam
        while True:
            success, frame = cap.read()
            if not success:
                print("Gagal menangkap frame.")
                break
            
            # Mengubah BGR ke RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Proses Wajah pada frame
            results = face_detection.process(img_rgb)
            # Menampilkan frame dengan bounding box
            cv2.imshow('Deteksi Wajah - Webcam (Bounding Box)', frame)

            if cv2.waitKey(1) & 0xFF == ord('b'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan deteksi wajah dari webcam
detect_faces_from_webcam()