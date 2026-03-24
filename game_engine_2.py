import math
import random
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QPolygonF
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from signals import bus

class ArenaWidget(QWidget): 
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(600, 400)
        
        # --- DIFFERENTIAL PHYSICS ---
        self.x = 80.0
        self.y = 100.0
        self.angle = 0.0  
        self.target_angle = 0.0
        self.turn_cooldown = 0
        
        self.cruise_active = False
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        
        self.v_left = 0.0
        self.v_right = 0.0
        self.wheel_base = 30.0 
        self.MAX_MOTOR_SPEED = 2.2 
        
        # --- ARENA GAME VARIABLES ---
        self.score = 0
        self.target_x = 0
        self.target_y = 0
        self.target_radius = 16
        self.walls = [] 
        
        self.finished = False 
        self.finish_rect = None 
        
        self.current_cmd = "STOP"
        self.active_modality = ""  
        
        bus.command_received.connect(self.update_command)
        bus.modality_changed.connect(self.update_modality) 
        
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_physics)
        self.move_timer.start(16) 

    def update_command(self, cmd):
        if self.finished: return
        if cmd == "CRUISE_TOGGLE": self.cruise_active = not self.cruise_active
        else: self.current_cmd = cmd

    def update_modality(self, modality):
        self.active_modality = modality
        self.update()

    def reset_game(self):
        self.build_arena()
        self.x = 80.0 
        self.y = 100.0
        self.angle = 0.0
        self.target_angle = 0.0
        self.turn_cooldown = 0
        self.cruise_active = False
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        self.v_left = 0.0
        self.v_right = 0.0
        self.current_cmd = "STOP"
        
        self.finished = False 
        self.score = 0
        self.spawn_target()
        self.update()

    def build_arena(self):
        """Creates a perfectly clean 3-Barrier Slalom Track"""
        self.walls.clear()
        w = self.width() if self.width() > 100 else 1000
        h = self.height() if self.height() > 100 else 800
        t = 24 # Wall thickness
        
        # 1. FINISH ZONE (Top-Right)
        self.finish_rect = QRectF(w * 0.75 + t, t, w * 0.25 - (t*2), 150)
        
        # 2. Outer Boundary Walls
        self.walls.append(QRectF(0, 0, w, t)) 
        self.walls.append(QRectF(0, h-t, w, t)) 
        self.walls.append(QRectF(0, 0, t, h)) 
        self.walls.append(QRectF(w-t, 0, t, h)) 

        # 3. Slalom Barriers (Perfectly spaced at 25%, 50%, 75%)
        # Barrier 1: Left (Top-Down)
        self.walls.append(QRectF(w * 0.25, 0, t, h * 0.75))      
        # Barrier 2: Center (Bottom-Up)
        self.walls.append(QRectF(w * 0.50, h * 0.25, t, h * 0.75))  
        # Barrier 3: Right (Top-Down)
        self.walls.append(QRectF(w * 0.75, 0, t, h * 0.75))      

    def spawn_target(self):
        w = self.width() if self.width() > 100 else 1000
        h = self.height() if self.height() > 100 else 800
        margin = 40
        
        while True:
            tx = random.randint(margin, w - margin)
            ty = random.randint(margin, h - margin)
            
            target_rect = QRectF(tx - 25, ty - 25, 50, 50)
            if not any(wall.intersects(target_rect) for wall in self.walls):
                self.target_x = tx
                self.target_y = ty
                break

    def process_physics(self):
        if self.finished: 
            self.v_left, self.v_right = 0, 0
            self.update()
            return
            
        if self.turn_cooldown > 0: self.turn_cooldown -= 16  
            
        target_v_l = 0.0
        target_v_r = 0.0
        
        if self.current_cmd == "BACKWARD": self.cruise_active = False 
            
        if self.cruise_active or self.current_cmd == "FORWARD":
            target_v_l = self.MAX_MOTOR_SPEED
            target_v_r = self.MAX_MOTOR_SPEED
        elif self.current_cmd == "BACKWARD":
            target_v_l = -self.MAX_MOTOR_SPEED
            target_v_r = -self.MAX_MOTOR_SPEED

        if self.current_cmd == "LEFT" and self.turn_cooldown <= 0:
            self.target_angle += 15.0; self.turn_cooldown = 1000  
        elif self.current_cmd == "RIGHT" and self.turn_cooldown <= 0:
            self.target_angle -= 15.0; self.turn_cooldown = 1000  

        # --- SMOOTH KINEMATICS & INERTIA ---
        angle_diff = self.target_angle - self.angle
        angle_diff = (angle_diff + 180) % 360 - 180

        if abs(angle_diff) > 0.5:
            turn_speed = self.MAX_MOTOR_SPEED * 0.4
            if angle_diff > 0: target_v_l = -turn_speed; target_v_r = turn_speed
            else: target_v_l = turn_speed; target_v_r = -turn_speed
        else:
            self.angle = self.target_angle
            
        angle_diff_phys = self.target_angle - self.angle
        angle_diff_phys = (angle_diff_phys + 180) % 360 - 180
        
        if abs(angle_diff_phys) > 0.5: accel = 0.025 
        elif target_v_l == 0 and target_v_r == 0: accel = 0.04 
        else: accel = 0.03 
            
        self.v_left += (target_v_l - self.v_left) * accel
        self.v_right += (target_v_r - self.v_right) * accel
        
        if abs(self.v_left) < 0.005: self.v_left = 0.0
        if abs(self.v_right) < 0.005: self.v_right = 0.0
        
        V = (self.v_right + self.v_left) / 2.0
        omega = (self.v_right - self.v_left) / self.wheel_base
        
        if abs(angle_diff_phys) > 0.5: self.angle += math.degrees(omega)
            
        rad = math.radians(self.angle)
        new_x = self.x + math.cos(rad) * V
        new_y = self.y - math.sin(rad) * V
        
        w, h = self.width(), self.height()
        new_x = max(20, min(new_x, w - 20))
        new_y = max(20, min(new_y, h - 20))

        # --- DETECT FINISH LINE ---
        chair_rect = QRectF(new_x - 18, new_y - 18, 36, 36)
        # ADDED 'self.finish_rect and' to prevent crashes before the map loads!
        if self.finish_rect and self.finish_rect.contains(chair_rect.center()):
            self.finished = True
            self.v_left, self.v_right = 0, 0 
            self.update()
            return

        # --- COLLISION DETECTION ---
        hit_wall = any(wall.intersects(chair_rect) for wall in self.walls)

        if not hit_wall:
            self.x = new_x
            self.y = new_y
            self.wheel_anim_left = (self.wheel_anim_left + self.v_left * 1.5) % 35
            self.wheel_anim_right = (self.wheel_anim_right + self.v_right * 1.5) % 35
        else:
            self.v_left, self.v_right = 0, 0 
            self.cruise_active = False
            # ==========================================
            # --- NEW: BROADCAST THE CRASH ---
            # ==========================================
            from signals import bus  # Fallback import just in case!
            bus.collision_occurred.emit("Wall / Boundary")

        dist = math.hypot(self.x - self.target_x, self.y - self.target_y)
        if dist < (18 + self.target_radius): 
            self.score += 10
            self.spawn_target()
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        if not self.walls: self.build_arena()
        t = 24 

        # 1. DRAW CONCRETE FACTORY FLOOR
        painter.fillRect(0, 0, w, h, QColor(50, 55, 60))
        painter.setPen(QPen(QColor(70, 75, 85), 1))
        for i in range(0, w, 50): painter.drawLine(i, 0, i, h)
        for i in range(0, h, 50): painter.drawLine(0, i, w, i)

        # 2. DRAW START / FINISH ZONES
        painter.setPen(QColor(150, 160, 170))
        font = painter.font()
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)

        # START: Top-Left
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 255, 100, 30))
        painter.drawRect(QRectF(t, t, w * 0.25 - t, 150))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(QRectF(t, t, w * 0.25 - t, 150), Qt.AlignCenter, "START")

        # FINISH: Top-Right
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 50, 50, 30))
        painter.drawRect(self.finish_rect) 
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(self.finish_rect, Qt.AlignCenter, "FINISH")

        # 3. DRAW INDUSTRIAL WALLS
        painter.setPen(QPen(QColor(255, 140, 0), 2)) # Safety Orange
        wall_grad = QLinearGradient(0, 0, 40, 40)
        wall_grad.setColorAt(0, QColor(40, 40, 40))
        wall_grad.setColorAt(1, QColor(20, 20, 20))
        painter.setBrush(wall_grad)
        for wall in self.walls: painter.drawRect(wall)

        # 4. DRAW TARGET COIN
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 215, 0, 80)) 
        painter.drawEllipse(QPointF(self.target_x, self.target_y), self.target_radius + 8, self.target_radius + 8)
        painter.setBrush(QColor(255, 215, 0)) 
        painter.drawEllipse(QPointF(self.target_x, self.target_y), self.target_radius, self.target_radius)
        painter.setPen(QPen(QColor(180, 150, 0), 3)) 
        painter.drawLine(QPointF(self.target_x - 8, self.target_y), QPointF(self.target_x + 8, self.target_y))
        painter.drawLine(QPointF(self.target_x, self.target_y - 8), QPointF(self.target_x, self.target_y + 8))

        # 5. DRAW SCOREBOARD
        painter.setPen(QColor(255, 255, 255))
        font.setPointSize(22)
        painter.setFont(font)
        y_offset = 70 if self.active_modality == "Voice Commands" else 40
        painter.drawText(20, y_offset, f"SCORE: {self.score}")

        # 6. DRAW 3D WHEELCHAIR
        painter.save()
        painter.translate(self.x, self.y)
        painter.rotate(-self.angle)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 90)) 
        painter.drawRoundedRect(QRectF(-22, -22, 55, 45), 5, 5)

        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(QColor(10, 10, 10), 2))
        painter.drawRoundedRect(QRectF(-20, -24, 35, 8), 3, 3) 
        painter.drawRoundedRect(QRectF(-20, 16, 35, 8), 3, 3)  
        
        painter.setPen(QPen(QColor(60, 60, 60), 2)) 
        for step in range(0, 36, 7):
            tread_x = -20 + ((step + self.wheel_anim_left) % 35)
            painter.drawLine(QPointF(tread_x, -24), QPointF(tread_x, -16))
            tread_x_r = -20 + ((step + self.wheel_anim_right) % 35)
            painter.drawLine(QPointF(tread_x_r, 16), QPointF(tread_x_r, 24))

        painter.setBrush(QColor(100, 100, 100))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(-8, -22, 4, 4))
        painter.drawEllipse(QRectF(-8, 18, 4, 4))

        painter.setBrush(QColor(80, 85, 90)) 
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawRect(QRectF(-10, -15, 25, 30))

        painter.setBrush(QColor(40, 40, 40))
        painter.drawRoundedRect(QRectF(16, -18, 10, 5), 2, 2) 
        painter.drawRoundedRect(QRectF(16, 13, 10, 5), 2, 2)  

        seat_gradient = QLinearGradient(QPointF(-10, -10), QPointF(10, 10))
        seat_gradient.setColorAt(0.0, QColor(30, 144, 255)) 
        seat_gradient.setColorAt(1.0, QColor(0, 50, 150))   
        painter.setBrush(seat_gradient)
        painter.setPen(QPen(QColor(0, 30, 100), 1))
        painter.drawRoundedRect(QRectF(-12, -12, 22, 24), 4, 4)

        backrest_grad = QLinearGradient(QPointF(-16, 0), QPointF(-10, 0))
        backrest_grad.setColorAt(0.0, QColor(20, 20, 20))
        backrest_grad.setColorAt(1.0, QColor(60, 60, 60))
        painter.setBrush(backrest_grad)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(-16, -12, 6, 24), 2, 2)

        painter.setBrush(QColor(180, 180, 190)) 
        painter.setPen(QPen(QColor(10, 10, 10), 1))
        painter.drawRoundedRect(QRectF(16, -8, 6, 16), 1, 1)

        painter.setBrush(QColor(0, 255, 0, 200))
        painter.setPen(Qt.NoPen)
        arrow = QPolygonF([QPointF(24, -4), QPointF(24, 4), QPointF(32, 0)])
        painter.drawPolygon(arrow)
        painter.restore()
        
        # 7. DRAW VOICE HUD
        if self.active_modality == "Voice Commands":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(10, 15, 20, 230))
            painter.drawRect(0, 0, w, 30)
            painter.setPen(QColor(255, 255, 255))
            font.setPointSize(11)
            painter.setFont(font)
            top_text = "🎙️ VOICE MODE: Say 'Rio start my wheelchair' to unlock." if self.current_cmd == "LOCK" else "🎙️ VOICE MODE: Say 'Rio lock my wheelchair' to lock."
            painter.drawText(QRectF(0, 0, w, 30), Qt.AlignCenter, top_text)
            painter.drawRect(0, h - 30, w, 30)
            painter.setPen(QColor(0, 255, 255)) 
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(QRectF(0, h - 30, w, 30), Qt.AlignCenter, "Command Format: 'Rio forward', 'Rio left', 'Rio stop'")

        # 8. DRAW "TEST PASSED" CARD
        if self.finished:
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 180))
            card_w, card_h = 450, 250
            card_x = int((w - card_w) / 2)
            card_y = int((h - card_h) / 2)
            card_rect = QRectF(card_x, card_y, card_w, card_h)
            
            painter.setPen(QPen(QColor(0, 255, 100, 200), 5))
            painter.setBrush(QBrush(QColor(10, 40, 20, 240))) 
            painter.drawRoundedRect(card_rect, 15, 15)
            
            font.setBold(True)
            font.setPointSize(28)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRectF(card_x, card_y + 40, card_w, 60), Qt.AlignCenter, "Test")
            
            painter.setPen(QPen(QColor(100, 110, 120), 2))
            painter.drawLine(card_x + 50, card_y + 105, card_x + card_w - 50, card_y + 105)
            
            font.setPointSize(22)
            painter.setFont(font)
            painter.setPen(QColor(0, 255, 100)) 
            painter.drawText(QRectF(card_x, card_y + 120, card_w, 60), Qt.AlignCenter, "STATUS: PASSED")
            
            font.setPointSize(18)
            painter.setFont(font)
            painter.setPen(QColor(200, 210, 220))
            painter.drawText(QRectF(card_x, card_y + 180, card_w, 60), Qt.AlignCenter, f"Final Score: {self.score} (Targets Collected)")