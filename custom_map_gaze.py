import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
class CustomGazeWorker(QThread):
    # Sends X, Y coordinates to the game engine
    gaze_coords = pyqtSignal(int, int)  
    status_update = pyqtSignal(str)
    frame_processed = pyqtSignal(QImage)
    
    
    def __init__(self, screen_w=1000, screen_h=800):
        super().__init__()
        self.running = True
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Mediapipe FaceMesh setup (Refined for Iris tracking)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # CRITICAL: This turns on the Iris tracking!
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Smoothing variables to prevent cursor jitter
        self.smooth_x = screen_w // 2
        self.smooth_y = screen_h // 2

    def run(self):
        cap = cv2.VideoCapture(0)
        self.status_update.emit("Webcam Active: Calibrating Gaze...")
        
        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            
            # Flip frame horizontally for a selfie-view mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # ==========================================
                # --- EYE MATH (Right Eye) ---
                # ==========================================
                r_eye_left = landmarks[33].x
                r_eye_right = landmarks[133].x
                r_eye_top = landmarks[159].y
                r_eye_bottom = landmarks[145].y
                r_iris_x = landmarks[468].x
                r_iris_y = landmarks[468].y
                
                r_ratio_x = (r_iris_x - r_eye_left) / (r_eye_right - r_eye_left + 1e-6)
                r_ratio_y = (r_iris_y - r_eye_top) / (r_eye_bottom - r_eye_top + 1e-6)

                # ==========================================
                # --- EYE MATH (Left Eye) ---
                # ==========================================
                l_eye_left = landmarks[362].x
                l_eye_right = landmarks[263].x
                l_eye_top = landmarks[386].y
                l_eye_bottom = landmarks[374].y
                l_iris_x = landmarks[473].x
                l_iris_y = landmarks[473].y
                
                l_ratio_x = (l_iris_x - l_eye_left) / (l_eye_right - l_eye_left + 1e-6)
                l_ratio_y = (l_iris_y - l_eye_top) / (l_eye_bottom - l_eye_top + 1e-6)
                
                # ==========================================
                # --- FUSE BOTH EYES FOR MAX STABILITY ---
                # ==========================================
                ratio_x = (r_ratio_x + l_ratio_x) / 2.0
                ratio_y = (r_ratio_y + l_ratio_y) / 2.0
                
                # Clamp the ratios to prevent the cursor from flying off screen
                ratio_x = max(0.0, min(1.0, ratio_x))
                ratio_y = max(0.0, min(1.0, ratio_y))
                
                # Map ratios to the 1000x800 map size
                target_x = int(ratio_x * self.screen_w)
                target_y = int(ratio_y * self.screen_h)
                
                # --- SMOOTHING FILTER ---
                self.smooth_x = int((0.8 * self.smooth_x) + (0.2 * target_x))
                self.smooth_y = int((0.8 * self.smooth_y) + (0.2 * target_y))
                
                # --- VISUAL PREVIEW: DRAW DOTS ON BOTH EYES ---
                h, w, ch = rgb_frame.shape
                cv2.circle(rgb_frame, (int(r_iris_x * w), int(r_iris_y * h)), 4, (0, 255, 255), -1)
                cv2.circle(rgb_frame, (int(l_iris_x * w), int(l_iris_y * h)), 4, (0, 255, 255), -1)
                
                # --- BROADCAST THE VIDEO TO MAIN UI ---
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_processed.emit(qt_image)
                
                # Emit the clean coordinates to game_engine_4.py!
                self.gaze_coords.emit(self.smooth_x, self.smooth_y)











        cap.release()
        self.face_mesh.close()
        self.status_update.emit("Gaze Tracking Offline")