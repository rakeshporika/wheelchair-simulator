import cv2
import mediapipe as mp
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QImage
from signals import bus

class VisionWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        
        # Cruise Control States
        self.cruise_hover_timer = 0
        self.hover_locked = False
        self.is_cruise_green = False

        # --- THE FIX: TIME-LOCKED ANCHORS ---
        self.anchor_x = 0.50
        self.anchor_y = 0.50
        self.is_anchored = False
        self.stability_timer = 0     # 1-second lock timer
        self.relocate_timer = 0      # 0.5-second move timer
        
        self.last_logged_cmd = ""    # <--- NEW: Added this to prevent CSV spam!

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(refine_landmarks=False)
        cap = cv2.VideoCapture(0)
        current_cmd = "STOP"
        
        # Dynamic Tuning (Adjust these to make the box bigger/smaller)
        SENSITIVITY_X = 0.25  
        SENSITIVITY_Y = 0.18  
        
        box_tl, box_br = (0, 0), (0, 0)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            h, w, ch = frame.shape 
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # ==========================================
                # --- NEW: CALCULATE HEAD SIZE IN PIXELS ---
                # ==========================================
                # Landmark 234 is the left cheekbone, 454 is the right cheekbone
                left_cheek = landmarks[234]
                right_cheek = landmarks[454]
                # Multiply by frame width (w) to get actual pixel distance
                head_size_px = int(abs(right_cheek.x - left_cheek.x) * w)
                
                nose = landmarks[1]
                
                bus.gaze_tracked.emit(nose.x, nose.y)
                
                # 1. Face Size & Center
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                face_width = max(x_coords) - min(x_coords)
                face_height = max(y_coords) - min(y_coords)
                
                face_center_x = (landmarks[234].x + landmarks[454].x) / 2.0
                face_center_y = (landmarks[10].y + landmarks[152].y) / 2.0
                
                # --- 2. TIME-LOCKED ANCHOR LOGIC ---
                if not self.is_anchored:
                    # STATE 1: Analyzing and locking onto face
                    self.anchor_x = face_center_x
                    self.anchor_y = face_center_y
                    self.stability_timer += 30 # Approx 30ms per frame
                    
                    if self.stability_timer >= 1000: # 1 second passed
                        self.is_anchored = True
                        self.stability_timer = 0
                        self.relocate_timer = 0
                else:
                    # STATE 2: Locked! Box will not move during head tilts.
                    # Check if user moved their whole body (shifted > 12% of screen)
                    dx = abs(face_center_x - self.anchor_x)
                    dy = abs(face_center_y - self.anchor_y)
                    
                    if dx > 0.12 or dy > 0.12:
                        self.relocate_timer += 30
                        if self.relocate_timer >= 500: # 0.5 seconds passed in new spot
                            self.is_anchored = False # Unlock to relocate!
                            self.relocate_timer = 0
                    else:
                        self.relocate_timer = 0 # Cancel relocation if they just peeked outside
                
                # 3. Calculate Box around the LOCKED anchor
                DEADZONE_X_MIN = self.anchor_x - (face_width * SENSITIVITY_X)
                DEADZONE_X_MAX = self.anchor_x + (face_width * SENSITIVITY_X)
                DEADZONE_Y_MIN = self.anchor_y - (face_height * SENSITIVITY_Y)
                DEADZONE_Y_MAX = self.anchor_y + (face_height * SENSITIVITY_Y)
                
                box_tl = (int(w * DEADZONE_X_MIN), int(h * DEADZONE_Y_MIN)) 
                box_br = (int(w * DEADZONE_X_MAX), int(h * DEADZONE_Y_MAX))
                
                nose_px = (int(nose.x * w), int(nose.y * h))
                cv2.circle(frame, nose_px, 6, (0, 0, 255), -1)
                
                # 4. Cruise Control Button
                # 4. Cruise Control Button
                btn_w, btn_h = 140, 45  # <--- INCREASED SIZE (was 100, 30)
                btn_x = int(w * self.anchor_x) - (btn_w // 2)
                
                # <--- PUSHED IT HIGHER (Changed - 10 to - 45)
                btn_y = box_tl[1] - btn_h - 45 
                
                if btn_y < 0: btn_y = 0

                if btn_x <= nose_px[0] <= btn_x + btn_w and btn_y <= nose_px[1] <= btn_y + btn_h:
                    if not self.hover_locked:
                        self.cruise_hover_timer += 30 
                        if self.cruise_hover_timer >= 1000: 
                            self.is_cruise_green = not self.is_cruise_green
                            self.hover_locked = True
                            self.cruise_hover_timer = 0
                            bus.command_received.emit("CRUISE_TOGGLE")
                else:
                    self.cruise_hover_timer = 0
                    self.hover_locked = False
                
                # 5. Direction Logic
                # Only trigger driving commands if we are fully anchored
                if self.is_anchored:
                    if nose.x < DEADZONE_X_MIN: current_cmd = "LEFT"
                    elif nose.x > DEADZONE_X_MAX: current_cmd = "RIGHT"
                    elif nose.y < DEADZONE_Y_MIN: current_cmd = "FORWARD"
                    elif nose.y > DEADZONE_Y_MAX: current_cmd = "BACKWARD"
                    else: current_cmd = "STOP"
                else:
                    current_cmd = "STOP" # Force stop while relocating for safety
                
                if current_cmd == "BACKWARD":
                    self.is_cruise_green = False
                
                bus.command_received.emit(current_cmd)
                
                # ==========================================
                # --- NEW: LOG THE HEAD METRICS TO CSV ---
                # ==========================================
                if current_cmd != self.last_logged_cmd:
                    self.last_logged_cmd = current_cmd
                    if current_cmd != "STOP":
                        bus.data_logged.emit("Head Tracking", f"Direction: {current_cmd} | Head Size: {head_size_px}px")

                # --- 6. DRAW VISUALS ---
                # Cruise Button
                btn_color = (0, 255, 0) if self.is_cruise_green else (100, 100, 100)
                overlay = frame.copy()
                cv2.rectangle(overlay, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), btn_color, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                if self.cruise_hover_timer > 0 and not self.hover_locked:
                    fill_pct = min(1.0, self.cruise_hover_timer / 1000.0)
                    fill_w = int(btn_w * fill_pct)
                    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + fill_w, btn_y + btn_h), (255, 200, 0), -1)
                    
                cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 2)
                #cv2.putText(frame, "CRUISE", (btn_x + 20, btn_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # We adjusted the X and Y coordinates (btn_x + 35, btn_y + 28) to center the text
                # We also bumped the font size up slightly from 0.5 to 0.6
                cv2.putText(frame, "CRUISE", (btn_x + 35, btn_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                current_cmd = "NO FACE"
                bus.command_received.emit("STOP")
                self.last_logged_cmd = "STOP"  # <--- NEW: Reset logger if face is lost
                box_tl, box_br = (0,0), (0,0)
                self.is_anchored = False # Reset lock if face lost

            # Draw the Tracking Box
            if box_tl != (0,0):
                if not self.is_anchored:
                    box_color = (0, 255, 255) # YELLOW when analyzing/relocating
                elif current_cmd == "STOP":
                    box_color = (255, 0, 0)   # BLUE when anchored & stopped
                else:
                    box_color = (0, 0, 255)   # RED when anchored & moving
                
                cv2.rectangle(frame, box_tl, box_br, box_color, 2)

            # Status Text
            status_text = "CALIBRATING..." if not self.is_anchored else f"CMD: {current_cmd}"
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255) if not self.is_anchored else ((0,0,255) if current_cmd != "STOP" else (255,0,0)), 3, cv2.LINE_AA)
            
            bytes_per_line = ch * w
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            bus.frame_updated.emit(qt_img.scaled(320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)) #change to 240 for previous view
            self.msleep(1) # Removed the 30ms artificial lag!
            
        cap.release()