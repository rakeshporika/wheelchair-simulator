import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QImage
from signals import bus
import math

class EyeTrackingWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.current_cmd = "STOP"
        
        # --- NEW: NOISE FILTER VARIABLES ---
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        # 80% old frame, 20% new frame. (Higher = smoother but slightly slower response)
        self.filter_strength = 0.80 

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            cmd = "STOP"
            
            # --- 1. DRAW HUD ZONES ---
            cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1) 
            cv2.rectangle(frame, (0, h-40), (w, h), (40, 40, 40), -1) 
            cv2.rectangle(frame, (0, 40), (60, h-40), (40, 40, 40), -1) 
            cv2.rectangle(frame, (w-60, 40), (w, h-40), (40, 40, 40), -1) 
            
            cv2.putText(frame, "FORWARD", (int(w/2)-50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            cv2.putText(frame, "BACK", (int(w/2)-30, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            cv2.putText(frame, "L", (20, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            cv2.putText(frame, "R", (w-40, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # --- 2. GET BOTH EYES ---
                # Left Eye (Appears on the right side of the screen)
                l_iris = landmarks[468]; l_inner = landmarks[133]; l_outer = landmarks[33]
                l_top = landmarks[159]; l_bottom = landmarks[145]
                
                # Right Eye (Appears on the left side of the screen)
                r_iris = landmarks[473]; r_inner = landmarks[362]; r_outer = landmarks[263]
                r_top = landmarks[386]; r_bottom = landmarks[374]
                
                # Convert irises to pixels for drawing
                l_ix, l_iy = int(l_iris.x * w), int(l_iris.y * h)
                r_ix, r_iy = int(r_iris.x * w), int(r_iris.y * h)
                
                # Draw green tracking dots on BOTH pupils
                cv2.circle(frame, (l_ix, l_iy), 4, (0, 255, 0), -1)
                cv2.circle(frame, (r_ix, r_iy), 4, (0, 255, 0), -1)

                # --- 3. CALCULATE STEREO GAZE RATIOS ---
                l_width = abs(l_outer.x - l_inner.x) * w
                l_height = abs(l_bottom.y - l_top.y) * h
                r_width = abs(r_outer.x - r_inner.x) * w
                r_height = abs(r_bottom.y - r_top.y) * h
                
                # BLINK PREVENTION: Only update if both eyes are actually open!
                if l_height > 3 and r_height > 3:
                    # Find the absolute left-most corner of each eye on the screen
                    l_leftmost_x = min(l_inner.x, l_outer.x) * w
                    r_leftmost_x = min(r_inner.x, r_outer.x) * w
                    
                    # Calculate raw ratios
                    raw_l_x = ((l_iris.x * w) - l_leftmost_x) / l_width
                    raw_l_y = ((l_iris.y * h) - (l_top.y * h)) / l_height
                    
                    raw_r_x = ((r_iris.x * w) - r_leftmost_x) / r_width
                    raw_r_y = ((r_iris.y * h) - (r_top.y * h)) / r_height
                    
                    # AVERAGE THEM TOGETHER (Halves the tracking error!)
                    raw_x = (raw_l_x + raw_r_x) / 2.0
                    raw_y = (raw_l_y + raw_r_y) / 2.0
                    
                    # --- 4. APPLY THE EMA NOISE FILTER ---
                    self.smooth_x = (self.filter_strength * self.smooth_x) + ((1 - self.filter_strength) * raw_x)
                    self.smooth_y = (self.filter_strength * self.smooth_y) + ((1 - self.filter_strength) * raw_y)
                    
                    
                # ==========================================
                # --- NEW: CALCULATE PUPIL ANGLE ---
                # ==========================================
                # We subtract 0.5 to find the offset from the dead-center of the eye
                dx = self.smooth_x - 0.5
                dy = self.smooth_y - 0.5
                # math.atan2 gives us the angle. We convert to degrees and force it between 0-360
                pupil_angle = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)


                # --- 5. DETERMINE DIRECTION USING SMOOTHED DATA ---
                if self.smooth_y < 0.35: 
                    cmd = "FORWARD"
                    cv2.rectangle(frame, (0, 0), (w, 40), (0, 255, 0), -1)
                    cv2.putText(frame, "FORWARD", (int(w/2)-50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                elif self.smooth_y > 0.65: 
                    cmd = "BACKWARD"
                    cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 255), -1)
                    cv2.putText(frame, "BACK", (int(w/2)-30, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                elif self.smooth_x < 0.40: 
                    cmd = "LEFT"
                    cv2.rectangle(frame, (0, 40), (60, h-40), (0, 255, 255), -1)
                    cv2.putText(frame, "L", (20, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                elif self.smooth_x > 0.60: 
                    cmd = "RIGHT"
                    cv2.rectangle(frame, (w-60, 40), (w, h-40), (0, 255, 255), -1)
                    cv2.putText(frame, "R", (w-40, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            if cmd != self.current_cmd:
                self.current_cmd = cmd
                bus.command_received.emit(cmd)
                
                # ==========================================
                # --- NEW: LOG THE EYE METRICS TO CSV ---
                # ==========================================
                if cmd != "STOP": # Only log active driving commands to keep the CSV clean
                    bus.data_logged.emit("Eye Tracking", f"Direction: {cmd} | Pupil Angle: {pupil_angle}°")
                    
            rgb_frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_frame_out.data, w, h, w * 3, QImage.Format_RGB888).copy()
            bus.frame_updated.emit(qt_img.scaled(320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.msleep(16)
            
        cap.release()