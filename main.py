import cv2
import numpy as np
import threading
import time
from face_detector import FaceDetector
from rppg_processor import rPPGProcessor
from respiration_processor import HeartRateProcessor
from signal_visualizer import RealtimeSignalGUI

class RealtimeHeartRateDetector:
    """Aplikasi utama untuk deteksi heart rate real-time dengan GUI"""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.rppg_processor = rPPGProcessor(fps=30, window_size=150)
        self.heart_rate_processor = HeartRateProcessor(fps=30)
        self.gui = RealtimeSignalGUI()
        
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        return False
        
    def stop_camera(self):
        """Hentikan kamera"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def process_frame(self, frame):
        """Proses frame untuk deteksi heart rate"""
        frame = cv2.flip(frame, 1)
        
        roi, bbox, confidence = self.face_detector.detect_face_roi(frame)
        
        if roi is not None and bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ROI Dahi", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            rgb_values, raw_signal, processed_signal = self.rppg_processor.process_frame(roi)
            
            if raw_signal is not None and processed_signal is not None:
                heart_rate = self.heart_rate_processor.calculate_heart_rate(processed_signal)
                
                self.gui.update_data(
                    raw_signal=raw_signal[-1] if len(raw_signal) > 0 else None,
                    processed_signal=processed_signal[-1] if len(processed_signal) > 0 else None,
                    heart_rate=heart_rate,
                    rgb_values=rgb_values
                )
                
                cv2.putText(frame, f"Heart Rate: {heart_rate:.1f} BPM", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                buffer_status = f"Buffer: {len(self.rppg_processor.rgb_buffer)}/150"
                cv2.putText(frame, buffer_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            else:
                if rgb_values is not None:
                    self.gui.update_data(rgb_values=rgb_values)
        else:
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
                
                cv2.imshow('Heart Rate Detection - Video Feed', processed_frame)
                
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
                print("Deteksi heart rate dimulai")
            else:
                print("Deteksi heart rate dihentikan")
        
        self.gui.toggle_detection = new_toggle
        
        try:
            self.gui.run()
        finally:
            self.stop_detection()

def main():
    """Fungsi utama"""
    print("=== Real-time Heart Rate Detection ===")
    print("Menggunakan metode rPPG dengan algoritma POS")
    print("Tekan 'Start Detection' untuk memulai")
    print("Tekan 'q' di window video untuk keluar")
    
    app = RealtimeHeartRateDetector()
    app.run()

if __name__ == "__main__":
    main()