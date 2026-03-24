import math
import random
import heapq  # <--- NEW: Python's Priority Queue for the A* Algorithm
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QPolygonF
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from signals import bus

class HomeArenaWidget(QWidget): 
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(600, 400)
        
        self.x = 200.0
        self.y = 200.0
        self.angle = 90.0  
        self.target_angle = 90.0
        self.turn_cooldown = 0
        
        self.cruise_active = False
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        self.v_left = 0.0
        self.v_right = 0.0
        self.wheel_base = 30.0 
        self.MAX_MOTOR_SPEED = 2.2  
        
        self.score = 0
        self.target_x = 0
        self.target_y = 0
        self.target_radius = 16
        self.walls = [] 
        
        self.autopilot_path = [] 
        
        self.current_cmd = "STOP"
        self.active_modality = ""  
        
        bus.command_received.connect(self.update_command)
        bus.modality_changed.connect(self.update_modality) 
        
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_physics)
        self.move_timer.start(16) 

    def update_command(self, cmd):
        nav_map = {
            "NAV_KITCHEN": "kitchen", "NAV_LIVING_ROOM": "living_mid",
            "NAV_GARAGE": "garage", "NAV_BATH": "bath",
            "NAV_GARDEN": "garden", "NAV_BEDROOM1": "bedroom1",
            "NAV_BEDROOM2": "bedroom2"
        }
        
        if cmd == "CRUISE_TOGGLE": 
            self.cruise_active = not self.cruise_active
            self.autopilot_path = [] 
        elif cmd in nav_map:
            self.calculate_path(nav_map[cmd])
        else: 
            self.current_cmd = cmd
            if cmd in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]:
                self.autopilot_path = [] 

    def update_modality(self, modality):
        self.active_modality = modality
        self.update()

    # ==========================================
    # --- ORTHOGONAL NAVMESH ---
    # ==========================================
    # ==========================================
    # --- ORTHOGONAL NAVMESH ---
    # ==========================================
    def define_navmesh(self):
        self.nav_nodes = {
            "garage": (0.3, 0.25), "garage_door": (0.3, 0.45), "living_left": (0.3, 0.55),
            "kitchen": (0.5, 0.25), "kitchen_door": (0.5, 0.45), "living_mid": (0.5, 0.55),
            "bed1_node": (0.6, 0.25), "bed1_door": (0.7, 0.25), "bedroom1": (0.82, 0.25), 
            "living_right": (0.6, 0.55), "hall_mid": (0.6, 0.50), "bath_door": (0.7, 0.50), "bath": (0.82, 0.50),
            "hall_bot": (0.6, 0.70), "bed2_door": (0.7, 0.70), "bedroom2": (0.82, 0.70),
            "garden_door": (0.5, 0.75), "garden": (0.5, 0.85)
        }
        
        self.nav_edges = {
            "garage": ["garage_door"], 
            "garage_door": ["garage", "living_left"], 
            "living_left": ["garage_door", "living_mid"],
            
            "kitchen": ["kitchen_door", "bed1_node"], 
            "kitchen_door": ["kitchen", "living_mid"], 
            
            "bed1_node": ["kitchen", "bed1_door"],
            "bed1_door": ["bed1_node", "bedroom1"],
            "bedroom1": ["bed1_door"],
            
            "living_mid": ["kitchen_door", "living_left", "living_right", "garden_door"],
            
            # --- FIXED: Properly chained the vertical hallway! ---
            "living_right": ["living_mid", "hall_mid", "hall_bot"], 
            
            "hall_mid": ["living_right", "bath_door"], 
            
            "bath_door": ["hall_mid", "bath"], 
            "bath": ["bath_door"],
            
            "hall_bot": ["living_right", "bed2_door"], 
            "bed2_door": ["hall_bot", "bedroom2"], 
            "bedroom2": ["bed2_door"],
            
            "garden_door": ["living_mid", "garden"], 
            "garden": ["garden_door"]
        }
    # ==========================================
    # --- NEW: A* (A-STAR) PATHFINDING ---
    # ==========================================
    def calculate_path(self, destination):
        self.define_navmesh()
        w, h = self.width(), self.height()
        
        # 1. Find the closest NavNode to the wheelchair's physical position
        start_node = None
        min_dist = float('inf')
        for node, (nx, ny) in self.nav_nodes.items():
            dist = math.hypot(self.x - (nx * w), self.y - (ny * h))
            if dist < min_dist:
                min_dist = dist
                start_node = node
                
        if not start_node or destination not in self.nav_nodes: return

        # 2. Heuristic: The straight-line pixel distance to the target
        def heuristic(node_a, node_b):
            ax, ay = self.nav_nodes[node_a]
            bx, by = self.nav_nodes[node_b]
            return math.hypot((ax - bx) * w, (ay - by) * h)

        # 3. Initialize A* Priority Queue
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        
        # Exact distance traveled so far
        g_score = {node: float('inf') for node in self.nav_nodes}
        g_score[start_node] = 0
        
        # Estimated total distance (traveled + heuristic)
        f_score = {node: float('inf') for node in self.nav_nodes}
        f_score[start_node] = heuristic(start_node, destination)

        while open_set:
            # Pop the node with the lowest f_score
            current = heapq.heappop(open_set)[1]

            # 4. Target Reached! Trace the path backward
            if current == destination:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                path.reverse() 
                
                # Convert the graph nodes into physical screen coordinates
                self.autopilot_path = [(self.nav_nodes[n][0] * w, self.nav_nodes[n][1] * h) for n in path]
                self.current_cmd = "AUTOPILOT"
                return

            # 5. Evaluate all connected neighbor nodes
            for neighbor in self.nav_edges.get(current, []):
                dist_to_neighbor = heuristic(current, neighbor)
                tentative_g_score = g_score[current] + dist_to_neighbor

                # If we found a faster, shorter route to this neighbor, record it!
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, destination)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    def reset_game(self):
        self.build_arena()
        w = self.width() if self.width() > 100 else 1000
        h = self.height() if self.height() > 100 else 800
        
        self.x = w * 0.3
        self.y = h * 0.6
        self.angle = 90.0
        self.target_angle = 90.0
        self.turn_cooldown = 0
        self.cruise_active = False
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        self.v_left = 0.0
        self.v_right = 0.0
        self.current_cmd = "STOP"
        self.autopilot_path = []
        
        self.score = 0
        self.spawn_target()
        self.update()

    def build_arena(self):
        self.walls.clear()
        w = self.width() if self.width() > 100 else 1000
        h = self.height() if self.height() > 100 else 800
        t = 16 
        
        self.walls.append(QRectF(w*0.05, h*0.05, w*0.9, t)) 
        self.walls.append(QRectF(w*0.05, h*0.05, t, h*0.7)) 
        self.walls.append(QRectF(w*0.95, h*0.05, t, h*0.7)) 
        self.walls.append(QRectF(w*0.05, h*0.75, w*0.4, t)) 
        self.walls.append(QRectF(w*0.55, h*0.75, w*0.4, t)) 
        self.walls.append(QRectF(w*0.35, h*0.05, t, h*0.4)) 
        self.walls.append(QRectF(w*0.05, h*0.45, w*0.2, t)) 
        self.walls.append(QRectF(w*0.35, h*0.45, w*0.1, t)) 
        self.walls.append(QRectF(w*0.55, h*0.45, w*0.15, t)) 
        self.walls.append(QRectF(w*0.70, h*0.05, t, h*0.15)) 
        self.walls.append(QRectF(w*0.70, h*0.30, t, h*0.15)) 
        self.walls.append(QRectF(w*0.70, h*0.55, t, h*0.10)) 
        self.walls.append(QRectF(w*0.70, h*0.35, w*0.25, t)) 
        self.walls.append(QRectF(w*0.70, h*0.55, w*0.25, t)) 

    def spawn_target(self):
        w = self.width() if self.width() > 100 else 1000
        h = self.height() if self.height() > 100 else 800
        while True:
            tx = random.randint(int(w*0.1), int(w*0.9))
            ty = random.randint(int(h*0.1), int(h*0.9))
            target_rect = QRectF(tx - 20, ty - 20, 40, 40)
            if not any(wall.intersects(target_rect) for wall in self.walls):
                self.target_x = tx
                self.target_y = ty
                break

    def process_physics(self):
        if self.turn_cooldown > 0: self.turn_cooldown -= 16  
            
        target_v_l = 0.0
        target_v_r = 0.0
        
        # ==========================================
        # --- BUTTER-SMOOTH DIFFERENTIAL AUTOPILOT ---
        # ==========================================
        if self.current_cmd == "AUTOPILOT" and self.autopilot_path:
            target_wp = self.autopilot_path[0]
            dist_to_wp = math.hypot(self.x - target_wp[0], self.y - target_wp[1])
            
            if dist_to_wp < 10: 
                self.autopilot_path.pop(0)
                if not self.autopilot_path:
                    self.current_cmd = "STOP" 
            else:
                desired_angle = math.degrees(math.atan2(-(target_wp[1] - self.y), target_wp[0] - self.x))
                self.target_angle = desired_angle
                
                angle_diff = self.target_angle - self.angle
                angle_diff = (angle_diff + 180) % 360 - 180
                
                # Sharp turns pivot smoothly
                if abs(angle_diff) > 12.0: 
                    turn_factor = min(0.6, max(0.15, abs(angle_diff) / 45.0))
                    turn_speed = self.MAX_MOTOR_SPEED * turn_factor 
                    
                    if angle_diff > 0:
                        target_v_l, target_v_r = -turn_speed, turn_speed
                    else:
                        target_v_l, target_v_r = turn_speed, -turn_speed
                # Micro-steering curves beautifully
                else: 
                    forward_speed = self.MAX_MOTOR_SPEED * 0.7 
                    correction = (angle_diff / 12.0) * (self.MAX_MOTOR_SPEED * 0.3)
                    
                    target_v_l = forward_speed - correction
                    target_v_r = forward_speed + correction
                    
        else:
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

            angle_diff = self.target_angle - self.angle
            angle_diff = (angle_diff + 180) % 360 - 180

            if abs(angle_diff) > 0.5:
                turn_speed = self.MAX_MOTOR_SPEED * 0.4
                if angle_diff > 0: target_v_l = -turn_speed; target_v_r = turn_speed
                else: target_v_l = turn_speed; target_v_r = -turn_speed
            else:
                self.angle = self.target_angle
                
        # --- SMOOTH PHYSICS APPLICATION ---
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

        chair_rect = QRectF(new_x - 16, new_y - 16, 32, 32)
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
            bus.collision_occurred.emit("House Wall")

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

        painter.fillRect(0, 0, w, h, QColor(30, 35, 40)) 
        painter.fillRect(QRectF(w*0.05, h*0.75, w*0.9, h*0.2), QColor(25, 50, 30)) 

        painter.setPen(QColor(100, 110, 120))
        font = painter.font()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        
        painter.drawText(QRectF(w*0.05, h*0.05, w*0.3, h*0.4), Qt.AlignCenter, "GARAGE")
        painter.drawText(QRectF(w*0.35, h*0.05, w*0.35, h*0.4), Qt.AlignCenter, "KITCHEN")
        painter.drawText(QRectF(w*0.05, h*0.45, w*0.65, h*0.3), Qt.AlignCenter, "LIVING ROOM")
        painter.drawText(QRectF(w*0.7, h*0.05, w*0.25, h*0.3), Qt.AlignCenter, "BEDROOM 1")
        painter.drawText(QRectF(w*0.7, h*0.35, w*0.25, h*0.2), Qt.AlignCenter, "BATH")
        painter.drawText(QRectF(w*0.7, h*0.55, w*0.25, h*0.2), Qt.AlignCenter, "BEDROOM 2")
        
        painter.setPen(QColor(80, 180, 80)) 
        painter.drawText(QRectF(w*0.05, h*0.75, w*0.9, h*0.2), Qt.AlignCenter, "GARDEN")

        if self.autopilot_path:
            painter.setPen(QPen(QColor(255, 100, 100, 150), 4, Qt.DashLine))
            current_pos = QPointF(self.x, self.y)
            for wp in self.autopilot_path:
                next_pos = QPointF(wp[0], wp[1])
                painter.drawLine(current_pos, next_pos)
                current_pos = next_pos
            painter.setBrush(QColor(255, 100, 100))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(current_pos, 8, 8)

        painter.setPen(QPen(QColor(0, 150, 255), 2))
        wall_grad = QLinearGradient(0, 0, 40, 40)
        wall_grad.setColorAt(0, QColor(50, 70, 90))
        wall_grad.setColorAt(1, QColor(30, 45, 60))
        painter.setBrush(wall_grad)
        for wall in self.walls: painter.drawRect(wall)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 215, 0, 80)) 
        painter.drawEllipse(QPointF(self.target_x, self.target_y), self.target_radius + 8, self.target_radius + 8)
        painter.setBrush(QColor(255, 215, 0)) 
        painter.drawEllipse(QPointF(self.target_x, self.target_y), self.target_radius, self.target_radius)
        painter.setPen(QPen(QColor(180, 150, 0), 3)) 
        painter.drawLine(QPointF(self.target_x - 8, self.target_y), QPointF(self.target_x + 8, self.target_y))
        painter.drawLine(QPointF(self.target_x, self.target_y - 8), QPointF(self.target_x, self.target_y + 8))

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

        seat_grad = QLinearGradient(QPointF(-10, -10), QPointF(10, 10))
        seat_grad.setColorAt(0.0, QColor(30, 144, 255)) 
        seat_grad.setColorAt(1.0, QColor(0, 50, 150))   
        painter.setBrush(seat_grad)
        painter.setPen(QPen(QColor(0, 30, 100), 1))
        painter.drawRoundedRect(QRectF(-12, -12, 22, 24), 4, 4)

        backrest_grad = QLinearGradient(QPointF(-16, 0), QPointF(-10, 0))
        backrest_grad.setColorAt(0.0, QColor(20, 20, 20))
        backrest_grad.setColorAt(1.0, QColor(60, 60, 60))
        painter.setBrush(backrest_grad)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(-16, -12, 6, 24), 2, 2)

        painter.setBrush(QColor(180, 180, 190)) 
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRoundedRect(QRectF(16, -8, 6, 16), 1, 1)
        painter.setBrush(QColor(0, 255, 0, 200))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(24, -4), QPointF(24, 4), QPointF(32, 0)]))
        painter.restore()

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
            painter.drawText(QRectF(0, h - 30, w, 30), Qt.AlignCenter, "Commands: 'Rio Kitchen', 'Rio Bedroom one', 'Rio forward', 'Rio stop'")
        
        painter.setPen(QColor(255, 255, 255))
        font.setPointSize(22)
        painter.setFont(font)
        painter.drawText(20, 40 if self.active_modality != "Voice Commands" else 70, f"SCORE: {self.score}")