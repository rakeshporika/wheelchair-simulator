import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import settings
from signals import bus

class GazeWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(settings.WEBCAM_ID)
        
        # Robust MediaPipe Face Mesh for precise nose/iris tracking
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Critical for Iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success: continue

                # Flip for mirror effect and convert color for MediaPipe
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = face_mesh.process(rgb_frame)
                h, w, _ = frame.shape
                
                current_cmd = "STOP"
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Get Nose Tip Coordinates (Landmark 1)
                    nose_x = int(landmarks[1].x * w)
                    nose_y = int(landmarks[1].y * h)
                    
                    # Draw a tracking dot on the nose
                    cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
                    
                    # --- GAZE-TO-UI HOVER LOGIC ---
                    # Divide the screen into a 3x3 grid. 
                    # If the nose moves into an outer zone, trigger the command!
                    margin_x, margin_y = w // 3, h // 3
                    
                    if nose_y < margin_y: current_cmd = "FORWARD"
                    elif nose_y > h - margin_y: current_cmd = "BACKWARD"
                    elif nose_x < margin_x: current_cmd = "LEFT"
                    elif nose_x > w - margin_x: current_cmd = "RIGHT"
                    
                # Broadcast the command to the physics engine
                bus.command_received.emit(current_cmd)
                
                # Send the camera frame to the PyQt5 GUI
                qt_img = self.convert_cv_qt(frame)
                bus.frame_updated.emit(qt_img)

        cap.release()

    def convert_cv_qt(self, cv_img):
        """Converts an OpenCV image to QImage for PyQt5 displaying."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        return QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()