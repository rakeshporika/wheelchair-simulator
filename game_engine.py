import math
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from signals import bus

class MazeWidget(QWidget): 
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        
        # --- DIFFERENTIAL DRIVE PHYSICS STATE ---
        self.x = 200.0
        self.y = 200.0
        self.angle = 90.0  
        
        # --- ADD THESE TWO LINES RIGHT HERE ---
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        # --------------------------------------
        
        # Stepped Turning State
        self.target_angle = 90.0
        self.turn_cooldown = 0
        
        # --- NEW: CRUISE CONTROL STATE ---
        self.cruise_active = False
        
        self.v_left = 0.0
        self.v_right = 0.0
        self.wheel_base = 30.0 
        
        self.MAX_MOTOR_SPEED = 2.0
        self.MOTOR_ACCEL = 0.15  
        
        self.current_cmd = "STOP"
        self.active_modality = ""  
        self.calib_stage = "DONE"
        
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        
        bus.gaze_tracked.connect(self.update_gaze)
        bus.command_received.connect(self.update_command)
        bus.modality_changed.connect(self.update_modality) 
        bus.calibration_step.connect(self.update_calib)
        
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_physics)
        self.move_timer.start(16) 

    def update_command(self, cmd):
        if cmd == "CRUISE_TOGGLE":
            self.cruise_active = not self.cruise_active
        else:
            self.current_cmd = cmd

    def update_modality(self, modality):
        self.active_modality = modality
        self.update()

    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y
        self.update()

    def update_calib(self, stage):
        self.calib_stage = stage
        self.update()

    def reset_game(self):
        self.x = self.width() / 2 if self.width() > 0 else 200.0
        self.y = self.height() / 2 if self.height() > 0 else 200.0
        self.angle = 90.0
        self.target_angle = 90.0
        self.turn_cooldown = 0
        #cruse control
        self.cruise_active = False
        self.cruise_hover_timer = 0
        self.hover_locked = False
        
        # --- NEW: TREAD ANIMATION TRACKERS ---
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        
        self.v_left = 0.0
        self.v_right = 0.0
        self.current_cmd = "STOP"
        self.update()
        
    def process_physics(self):
        w, h = self.width(), self.height()
        
        # 1. Calculate screen coordinates of your Nose/Eyes
        min_val, max_val = 0.35, 0.65 
        norm_x = max(0.0, min(1.0, (self.gaze_x - min_val) / (max_val - min_val)))
        norm_y = max(0.0, min(1.0, (self.gaze_y - min_val) / (max_val - min_val)))
        cx, cy = norm_x * w, norm_y * h
        
        # 2. Cruise Control Button Hover Logic
        btn_w, btn_h = 180, 50
        btn_x, btn_y = (w - btn_w) / 2, 5  # Top Center of the screen
        
        if btn_x <= cx <= btn_x + btn_w and btn_y <= cy <= btn_y + btn_h:
            if not self.hover_locked:
                self.cruise_hover_timer += 16 # Add 16 milliseconds
                if self.cruise_hover_timer >= 1200: # 1.2 seconds to toggle
                    self.cruise_active = not self.cruise_active # Toggle On/Off
                    self.hover_locked = True # Lock it so it doesn't instantly toggle again
                    self.cruise_hover_timer = 0
        else:
            # If you look away from the button, reset the timer and unlock it
            self.cruise_hover_timer = 0
            self.hover_locked = False

        # 3. Manage the 1-second Turn Halt Timer
        if self.turn_cooldown > 0:
            self.turn_cooldown -= 16  
            
        target_v_l = 0.0
        target_v_r = 0.0
        
        # --- 4. Standard Driving & Braking ---
        if self.current_cmd == "BACKWARD":
            self.cruise_active = False # Hitting the brakes cancels cruise control!
            
        if self.cruise_active:
            target_v_l = self.MAX_MOTOR_SPEED
            target_v_r = self.MAX_MOTOR_SPEED
        elif self.current_cmd == "FORWARD":
            target_v_l = self.MAX_MOTOR_SPEED
            target_v_r = self.MAX_MOTOR_SPEED
        elif self.current_cmd == "BACKWARD":
            target_v_l = -self.MAX_MOTOR_SPEED
            target_v_r = -self.MAX_MOTOR_SPEED

        # 5. Ratchet Turning (Overrides forward momentum temporarily)
        if self.current_cmd == "LEFT" and self.turn_cooldown <= 0:
            self.target_angle += 15.0
            self.turn_cooldown = 1000  
        elif self.current_cmd == "RIGHT" and self.turn_cooldown <= 0:
            self.target_angle -= 15.0
            self.turn_cooldown = 1000  

        # Auto-Rotate
        angle_diff = self.target_angle - self.angle
        angle_diff = (angle_diff + 180) % 360 - 180

        if abs(angle_diff) > 0.5:
            turn_speed = self.MAX_MOTOR_SPEED * 0.4 #change jerkings while turning
            if angle_diff > 0: 
                target_v_l = -turn_speed
                target_v_r = turn_speed
            else: 
                target_v_l = turn_speed
                target_v_r = -turn_speed
        else:
            self.angle = self.target_angle
            
        # Apply Momentum & Kinematics
        #self.v_left += (target_v_l - self.v_left) * self.MOTOR_ACCEL
        #self.v_right += (target_v_r - self.v_right) * self.MOTOR_ACCEL
        
        # --- NEW: REALISTIC WEIGHT & INERTIA ---
        if abs(angle_diff) > 0.5:
            # 1. High Torque: Fast pivoting
            accel = 0.05 #torque while turning
        elif target_v_l == 0 and target_v_r == 0:
            # 2. Coasting: Gradual stop/friction when you let go
            accel = 0.04 
        else:
            # 3. Heavy Acceleration: Slowly building up speed
            accel = 0.015 
            
        self.v_left += (target_v_l - self.v_left) * accel
        self.v_right += (target_v_r - self.v_right) * accel
        
        # Micro-drift cutoff: physically locks the brakes when moving incredibly slow
        if abs(self.v_left) < 0.02: self.v_left = 0.0
        if abs(self.v_right) < 0.02: self.v_right = 0.0
        #
        
        V = (self.v_right + self.v_left) / 2.0
        omega = (self.v_right - self.v_left) / self.wheel_base
        
        if abs(angle_diff) > 0.5:
            self.angle += math.degrees(omega)
            
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * V
        self.y -= math.sin(rad) * V
        
        self.x = max(20, min(self.x, w - 20))
        self.y = max(20, min(self.y, h - 20))
        # --- NEW: SPIN THE TIRE TREADS ---
        # We add the motor speed to the animation loop. 
        # Modulo 35 keeps the animation wrapping perfectly around the 35-pixel-long wheel!
        self.wheel_anim_left = (self.wheel_anim_left + self.v_left * 1.5) % 35
        self.wheel_anim_right = (self.wheel_anim_right + self.v_right * 1.5) % 35
        self.update()
        
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        #
        
        # --- Draw Floor & Grid ---
        painter.fillRect(self.rect(), QColor("#2c3e50")) 
        
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1)) 
        grid_size = 50
        for i in range(0, w, grid_size):
            painter.drawLine(i, 0, i, h)
        for i in range(0, h, grid_size):
            painter.drawLine(0, i, w, i)
        """  
        # --- Draw Wheelchair ---
        painter.translate(self.x, self.y)
        painter.rotate(-(self.angle - 90)) 
        
        painter.setBrush(QBrush(QColor("#3498db")))
        painter.setPen(QPen(QColor("#2980b9"), 2))
        body_rect = QRectF(-20, -20, 40, 40)
        painter.drawRoundedRect(body_rect, 5, 5)
        
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(-15, 10, 30, 5))
        
        painter.setBrush(QBrush(QColor("#111111")))
        painter.drawRoundedRect(QRectF(-25, -15, 5, 30), 2, 2) 
        painter.drawRoundedRect(QRectF(20, -15, 5, 30), 2, 2)  
        
        painter.setPen(QPen(QColor(255, 50, 50), 3))
        painter.drawLine(QPointF(0, 0), QPointF(0, -35))
        """    
        # ==========================================
        # --- 3D WHEELCHAIR DRAWING ---
        # ==========================================
        painter.save()
        painter.translate(self.x, self.y)
        
        # Qt's rotation is clockwise, but our math assumes standard Cartesian (Counter-Clockwise).
        # We rotate by -self.angle so it points the right way.
        painter.rotate(-self.angle)
        
        # 1. DROP SHADOW (Gives the illusion of floating/height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 90))
        painter.drawRoundedRect(QRectF(-22, -22, 55, 45), 5, 5)

        # 2. REAR DRIVE WHEELS (With Animated Treads)
        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(QColor(10, 10, 10), 2))
        
        # Draw Base Tires
        painter.drawRoundedRect(QRectF(-20, -24, 35, 8), 3, 3) # Left Wheel
        painter.drawRoundedRect(QRectF(-20, 16, 35, 8), 3, 3)  # Right Wheel
        
        # --- NEW: Draw Rolling Treads ---
        painter.setPen(QPen(QColor(60, 60, 60), 2)) # Lighter gray for treads
        
        # Left Tire Treads
        for step in range(0, 36, 7): # Space treads every 7 pixels
            tread_x = -20 + ((step + self.wheel_anim_left) % 35)
            painter.drawLine(QPointF(tread_x, -24), QPointF(tread_x, -16))
            
        # Right Tire Treads
        for step in range(0, 36, 7):
            tread_x = -20 + ((step + self.wheel_anim_right) % 35)
            painter.drawLine(QPointF(tread_x, 16), QPointF(tread_x, 24))

        # Hubcaps for Drive Wheels
        painter.setBrush(QColor(100, 100, 100))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(-8, -22, 4, 4))
        painter.drawEllipse(QRectF(-8, 18, 4, 4))
        # Hubcaps for Drive Wheels
        painter.setBrush(QColor(100, 100, 100))
        painter.drawEllipse(QRectF(-8, -22, 4, 4))
        painter.drawEllipse(QRectF(-8, 18, 4, 4))

        # 3. CHASSIS / METAL FRAME
        painter.setBrush(QColor(80, 85, 90)) 
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawRect(QRectF(-10, -15, 25, 30))

        # 4. FRONT CASTERS (Small turning wheels)
        painter.setBrush(QColor(40, 40, 40))
        painter.drawRoundedRect(QRectF(16, -18, 10, 5), 2, 2) # Left Caster
        painter.drawRoundedRect(QRectF(16, 13, 10, 5), 2, 2)  # Right Caster

        # 5. 3D SEAT CUSHION (Using a Gradient for depth)
        from PyQt5.QtGui import QLinearGradient
        seat_gradient = QLinearGradient(QPointF(-10, -10), QPointF(10, 10))
        seat_gradient.setColorAt(0.0, QColor(30, 144, 255)) # Light Blue highlight
        seat_gradient.setColorAt(1.0, QColor(0, 50, 150))   # Deep Blue shadow
        
        painter.setBrush(seat_gradient)
        painter.setPen(QPen(QColor(0, 30, 100), 1))
        painter.drawRoundedRect(QRectF(-12, -12, 22, 24), 4, 4)

        # 6. BACKREST (Thick shape at the rear)
        backrest_grad = QLinearGradient(QPointF(-16, 0), QPointF(-10, 0))
        backrest_grad.setColorAt(0.0, QColor(20, 20, 20))
        backrest_grad.setColorAt(1.0, QColor(60, 60, 60))
        painter.setBrush(backrest_grad)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(-16, -12, 6, 24), 2, 2)

        # 7. FOOTREST (Metallic plate at the front)
        painter.setBrush(QColor(180, 180, 190)) 
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRoundedRect(QRectF(16, -8, 6, 16), 1, 1)

        # 8. DIRECTIONAL GLOW (So you always know which way is forward)
        painter.setBrush(QColor(0, 255, 0, 200))
        painter.setPen(Qt.NoPen)
        from PyQt5.QtGui import QPolygonF
        arrow = QPolygonF([QPointF(24, -4), QPointF(24, 4), QPointF(32, 0)])
        painter.drawPolygon(arrow)

        painter.restore()
        # ==========================================
        # --- EYE TRACKING TRANSPARENT OVERLAY ---
        if "Eye Tracking" in self.active_modality:
            # Reset camera so the UI doesn't spin with the wheelchair
            painter.resetTransform() 
            
            font = painter.font()
            font.setPointSize(16)
            font.setBold(True)
            painter.setFont(font)
            
            # Top Zone (FORWARD)
            painter.fillRect(0, 0, w, 60, QColor(0, 255, 0, 30))
            painter.setPen(QColor(0, 255, 0, 150))
            painter.drawText(QRectF(0, 0, w, 60), Qt.AlignCenter, "FORWARD (Look Up)")
            
            # Bottom Zone (BACKWARD)
            painter.fillRect(0, h-60, w, 60, QColor(255, 0, 0, 30))
            painter.setPen(QColor(255, 0, 0, 150))
            painter.drawText(QRectF(0, h-60, w, 60), Qt.AlignCenter, "BACKWARD (Look Down)")
            
            # Left Zone (LEFT)
            painter.fillRect(0, 60, 60, h-120, QColor(0, 150, 255, 30))
            painter.setPen(QColor(0, 150, 255, 150))
            painter.drawText(QRectF(0, 60, 60, h-120), Qt.AlignCenter | Qt.TextWordWrap, "L\nE\nF\nT")
            
            # Right Zone (RIGHT)
            painter.fillRect(w-60, 60, 60, h-120, QColor(0, 150, 255, 30))
            painter.setPen(QColor(0, 150, 255, 150))
            painter.drawText(QRectF(w-60, 60, 60, h-120), Qt.AlignCenter | Qt.TextWordWrap, "R\nI\nG\nH\nT")
            
            # --- NEW: Center Zone (STOP) ---
            center_w = 180
            center_h = 60
            cx = int((w - center_w) / 2)
            cy = int((h - center_h) / 2)
            
            # Draw a subtle semi-transparent white box in the exact middle
            painter.fillRect(cx, cy, center_w, center_h, QColor(255, 255, 255, 20))
            
            # Draw the text and a dashed border to make it look like a resting zone
            painter.setPen(QPen(QColor(255, 255, 255, 150), 2, Qt.DashLine))
            painter.drawRoundedRect(cx, cy, center_w, center_h, 8, 8)
            
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawText(QRectF(cx, cy, center_w, center_h), Qt.AlignCenter, "STOP (Center)")
            
            # --- NEW: GAZE CURSOR (YELLOW TARGET) ---
            # The eye tracking values normally float between 0.35 and 0.65.
            # We map those values to stretch across the full width and height of the screen.
            min_val = 0.35
            max_val = 0.65
            span = max_val - min_val
            
            # Clamp the values so the yellow dot doesn't fly off the edge of the screen
            norm_x = max(0.0, min(1.0, (self.gaze_x - min_val) / span))
            norm_y = max(0.0, min(1.0, (self.gaze_y - min_val) / span))
            
            screen_x = norm_x * w
            screen_y = norm_y * h
            
            # Draw a bright yellow target circle with a red center dot
            painter.setPen(QPen(QColor(255, 255, 0, 200), 3))
            painter.setBrush(QColor(255, 255, 0, 100)) # Semi-transparent yellow fill
            painter.drawEllipse(QPointF(screen_x, screen_y), 18, 18)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0)) # Red center dot
            painter.drawEllipse(QPointF(screen_x, screen_y), 4, 4)
            
            # --- NEW: VOICE COMMANDS TRANSPARENT OVERLAY ---
        if "Voice" in self.active_modality:
            painter.resetTransform() 
            
        """ 
           #Draw Top Instruction Banner
            painter.fillRect(0, 0, w, 40, QColor(0, 0, 0, 180))
            font = painter.font()
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRectF(0, 0, w, 40), Qt.AlignCenter, "🎤 VOICE MODE: Say 'Rio start my wheelchair' to unlock.")
            # --- DYNAMIC VOICE HUD TEXT ---
        if self.active_modality == "Voice Commands":
            if self.current_cmd == "LOCK":
                status_msg = "🎤 VOICE MODE: Say 'Rio start my wheelchair' to unlock."
            else:
                status_msg = "🎤 VOICE MODE: Say 'Rio lock my wheelchair' to lock."
            
            # Put 'status_msg' inside your existing drawText function!
            # Example: painter.drawText(x, y, status_msg)
            # Draw Bottom Format Guide
            painter.fillRect(0, h-40, w, 40, QColor(0, 0, 0, 180))
            painter.setPen(QColor(0, 255, 255)) # Cyan text
            painter.drawText(QRectF(0, h-40, w, 40), Qt.AlignCenter, "Command Format: 'Rio forward', 'Rio left', 'Rio stop'")
            """
            
        # ==========================================
        # --- HUD (TOP AND BOTTOM BARS) ---
        # ==========================================
        # ONLY draw these instructional banners if Voice Commands are active!
        if self.active_modality == "Voice Commands":
            
            # 1. Top Bar Background
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(10, 15, 20, 230))
            painter.drawRect(0, 0, w, 30)

            # 2. Dynamic Top Bar Text
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)

            if self.current_cmd == "LOCK":
                top_text = "🎙️ VOICE MODE: Say 'Rio start my wheelchair' to unlock."
            else:
                top_text = "🎙️ VOICE MODE: Say 'Rio lock my wheelchair' to lock."

            painter.drawText(QRectF(0, 0, w, 30), Qt.AlignCenter, top_text)

            # 3. Bottom Bar Background & Text
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(10, 15, 20, 230))
            painter.drawRect(0, h - 30, w, 30)
            
            painter.setPen(QColor(0, 255, 255)) 
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(QRectF(0, h - 30, w, 30), Qt.AlignCenter, "Command Format: 'Rio forward', 'Rio left', 'Rio stop'")