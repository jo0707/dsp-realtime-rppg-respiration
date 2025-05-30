import cv2
import numpy as np
from threading import Thread
import time

class Camera:
    def __init__(self, src=0, fps=30):
        self.src = src
        self.fps = fps
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame = None
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = Thread(target=self._update)
        self.thread.start()
        
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            time.sleep(1.0 / self.fps)
            
    def get_frame(self):
        return self.frame
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()

def main_detection_loop():
    # This is kept for backward compatibility
    # The main application is now in shoulder_respiration_gui.py
    from shoulder_respiration_gui import main
    main()