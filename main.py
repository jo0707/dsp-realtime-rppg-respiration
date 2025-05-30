import sys
from PyQt5.QtWidgets import QApplication
from camera import CameraProcessor
from signal_processor import SignalProcessor
from gui import MainWindow

def main():
    """Main entry point of the application"""
    app = QApplication(sys.argv)
    
    # Initialize components
    fps = 20
    buffer_size = 200
    
    # Create processor instances
    camera_processor = CameraProcessor(fps=fps)
    signal_processor = SignalProcessor(fps=fps, buffer_size=buffer_size)
    
    # Create and show main window
    window = MainWindow(camera_processor, signal_processor)
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
