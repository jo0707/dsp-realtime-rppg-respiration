import cv2
import numpy as np
import threading
import time
from shoulder_detector import ShoulderMovementDetector
from shoulder_respiration_processor import ShoulderRespirationProcessor
from shoulder_respiration_gui import ShoulderRespirationGUI

class ShoulderRespirationDetector:
    """Aplikasi utama untuk deteksi respirasi berdasarkan pergerakan bahu"""
    
    def __init__(self):
        self.shoulder_detector = ShoulderMovementDetector()
        self.respiration_processor = ShoulderRespirationProcessor(fps=30, window_size=300)
        self.gui = ShoulderRespirationGUI()
        
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        
    def start_camera(self):
        """Inisialisasi kamera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  # Increased from 640 to 1000
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)  # Increased from 480 to 750
            return True
        return False
        
    def stop_camera(self):
        """Hentikan kamera"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def process_frame(self, frame):
        """Proses frame untuk deteksi respirasi dari bahu"""
        frame = cv2.flip(frame, 1)
        
        # Detect shoulder landmarks
        left_pos, right_pos, pose_landmarks = self.shoulder_detector.detect_shoulder_landmarks(frame)
        
        if left_pos and right_pos:
            # Draw shoulder landmarks
            frame = self.shoulder_detector.draw_shoulder_landmarks(frame, left_pos, right_pos, pose_landmarks)
            
            # Calculate shoulder movement
            shoulder_data = self.shoulder_detector.calculate_shoulder_movement(left_pos, right_pos)
            
            if shoulder_data:
                # Process movement for respiration
                raw_signal, filtered_signal, respiration_rate = self.respiration_processor.process_movement(shoulder_data)
                
                # Update GUI
                self.gui.update_data(
                    raw_signal=raw_signal,
                    filtered_signal=filtered_signal,
                    respiration_rate=respiration_rate,
                    shoulder_data=shoulder_data
                )
                
                # Display info on video
                cv2.putText(frame, f"Respirasi: {respiration_rate:.1f} BPM", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                buffer_status = f"Buffer: {len(self.respiration_processor.movement_buffer)}/300"
                cv2.putText(frame, buffer_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                # Display shoulder center position
                center_y = shoulder_data['center_y']
                cv2.putText(frame, f"Center Y: {center_y:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Posisi badan tidak terdeteksi", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Berdiri tegak di depan kamera", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return frame
    
    def detection_loop(self):
        """Loop utama untuk deteksi real-time"""
        if not self.start_camera():
            print("Error: Tidak dapat mengakses webcam")
            return
            
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.gui.is_recording:
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = cv2.flip(frame, 1)
                    cv2.putText(processed_frame, "Tekan 'Start Detection' untuk mulai", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(processed_frame, "Berdiri tegak dengan bahu terlihat", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Shoulder Movement Respiration Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                time.sleep(1/30)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop_camera()
    
    def start_detection(self):
        """Mulai deteksi dalam thread terpisah"""
        if not self.is_running:
            self.is_running = True
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
    
    def stop_detection(self):
        """Hentikan deteksi"""
        self.is_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
    
    def run(self):
        """Jalankan aplikasi utama"""
        self.start_detection()
        
        original_toggle = self.gui.toggle_detection
        def new_toggle():
            original_toggle()
            if self.gui.is_recording:
                print("Deteksi respirasi dari bahu dimulai")
            else:
                print("Deteksi respirasi dihentikan")
        
        self.gui.toggle_detection = new_toggle
        
        try:
            self.gui.run()
        finally:
            self.stop_detection()

def main():
    """Fungsi utama"""
    print("=== Shoulder Movement Respiration Detection ===")
    print("Deteksi sinyal respirasi dari pergerakan bahu")
    print("Fitur:")
    print("- Tracking pergerakan bahu kiri dan kanan")
    print("- Analisis pergerakan vertikal untuk respirasi")
    print("- Visualisasi real-time dengan 4 grafik:")
    print("  1. Posisi Y bahu (kiri, kanan, center)")
    print("  2. Sinyal pergerakan mentah")
    print("  3. Sinyal respirasi yang difilter")
    print("  4. Breathing rate (BPM)")
    print("\nPetunjuk penggunaan:")
    print("- Berdiri tegak di depan kamera")
    print("- Pastikan bahu terlihat jelas")
    print("- Tekan 'Start Detection' untuk memulai")
    print("- Tekan 'q' di window video untuk keluar")
    
    app = ShoulderRespirationDetector()
    app.run()

if __name__ == "__main__":
    main()