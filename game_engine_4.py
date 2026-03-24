import os
import json
import math
import heapq
import cv2
import numpy as np
import torch
import torch.nn as nn
import helper
import subprocess
import platform # <--- NEW: Allows Python to detect the OS
#import pyttsx3
import threading
import time
# ==========================================
# --- ENVIRONMENT SETUP ---
# ==========================================
# Resolves a known macOS issue where PyTorch and PyQt5 conflict over threading
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from PyQt5.QtWidgets import (QWidget, QFileDialog, QInputDialog, 
                             QMessageBox, QPushButton, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import (QPainter, QColor, QPen, QImage, QFont, 
                         QPolygonF, QLinearGradient)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF

# Custom event bus for handling Voice/EEG signals across the app
from signals import bus

# ==========================================
# --- NEURAL NETWORK ARCHITECTURE ---
# ==========================================
class DQN(nn.Module):
    """
    Deep Q-Network for Reinforcement Learning (Experimental).
    Takes 7 inputs (sensors + target angle) and outputs 3 actions (Left, Right, Forward).
    """
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# --- MAIN GAME ENGINE WIDGET ---
# ==========================================
class DynamicHomeWidget(QWidget):
    """
    The core 2D Physics and Pathfinding Simulator for the Wheelchair.
    Handles map loading, computer vision, A* pathing, LiDAR, and differential kinematics.
    """
    def __init__(self):
        super().__init__()
        
        # --- UI & Core Dimensions ---
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(1000, 800)
        self.map_w, self.map_h = 1000, 800 
        
        # --- Wheelchair Kinematics State ---
        self.x, self.y = 100.0, 100.0
        self.angle = 90.0  
        self.target_angle = 90.0
        self.turn_cooldown = 0
        self.cruise_active = False
        self.wheel_anim_left = 0.0
        self.wheel_anim_right = 0.0
        self.v_left = 0.0
        self.v_right = 0.0
        self.wheel_base = 30.0 
        self.MAX_MOTOR_SPEED = 1.8
        self.current_cmd = "STOP"
       
       # ==========================================
        # --- DIGITAL TWIN: REAL-WORLD SCALING ---
        # ==========================================
        self.real_world_width_ft = 50.0  
        self.real_world_length_ft = 40.0 
        self.pixels_per_foot = 20.0
        self.pixels_per_inch = self.pixels_per_foot / 12.0
        
        # PHYSICAL PROTOTYPE DIMENSIONS (6-inch Circular Robot)
        self.prototype_diameter_inches = 6.0 
        self.robot_pixel_radius = 5.0 # Will be dynamically calculated
        self.prototype_speed_inches_per_sec = 6.0 #change speed as per user.
        
         
        # --- Map & Memory State ---
        self.config_file = "custom_map_config.json"
        self.map_image_path = ""
        self.map_image = None
        self.grid = None           
        self.path_grid = None # Fat grid for safe A* padding
        self.yolo_model = helper.load_yolo_model()
        self.grid_scale = 10       
        self.rooms = {} # Stores voice command targets (Custom + YOLO)           
        self.autopilot_path = []
        self.active_modality = ""
        self.edit_mode = False 

        # ==========================================
        # --- EYE TRACKING STATE (SIMULATED VIA MOUSE) ---
        # ==========================================
        self.setMouseTracking(True) # Allows reading mouse without clicking
        self.gaze_x = -100.0
        self.gaze_y = -100.0
        self.dwell_target = None
        self.dwell_progress = 0.0  
        self.DWELL_THRESHOLD = 90  # 1.5 seconds at 60 frames per second
        
        
        
        # --- LiDAR Sensor Array ---
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensor_distances = [100] * 5
        
        # --- Initialization ---
        self.setup_ui()
        
        # Attempt to load RL brain (Fallback to pure math if missing)
        self.rl_agent = DQN(7, 3)
        try:
            self.rl_agent.load_state_dict(torch.load("wheelchair_rl_brain.pth"))
            self.rl_agent.eval() 
            self.rl_enabled = True
            print("SUCCESS: PyTorch RL Brain Loaded!")
        except Exception as e:
            print("No RL brain found, defaulting to standard Autopilot.")
            self.rl_enabled = False
        
        # Load previous map session
        self.load_config()

        # Connect Signal Bus for Voice/GUI commands
        bus.command_received.connect(self.update_command)
        bus.modality_changed.connect(self.update_modality)
        bus.command_received.connect(self.handle_voice_navigation)
        
        # Start Physics Loop (Runs 60 times a second)
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_physics)
        self.move_timer.start(16)


    # ==========================================
    # --- GUI & MEMORY MANAGEMENT ---
    # ==========================================
    def setup_ui(self):
        """Builds the overlay buttons for Map Uploading and Room Editing."""
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        
        self.btn_upload = QPushButton("Upload New Floorplan")
        self.btn_upload.setStyleSheet("background-color: #3498db; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        self.btn_upload.clicked.connect(self.upload_new_map)
        
        self.btn_edit = QPushButton("Edit Room Labels")
        self.btn_edit.setStyleSheet("background-color: #e67e22; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        self.btn_edit.clicked.connect(self.toggle_edit_mode)
        
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addStretch()

    def load_config(self):
        """Loads the saved map path and voice dictionary, then reconstructs the environment."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                    self.map_image_path = data.get("map_path", "")
                    self.rooms = data.get("rooms", {})
                    
                    if self.map_image_path and os.path.exists(self.map_image_path):
                        print(f"📂 Restoring saved floorplan: {self.map_image_path}")
                        self.process_image(self.map_image_path)
                        self.run_yolo_on_map(self.map_image_path) 
            except Exception as e:
                print(f"Error loading config: {e}")
                
    #code for saving only useer entered labels
    def save_config(self):
        """Saves ONLY Custom Pins into the config file. YOLO is handled dynamically."""
        # We only save self.rooms (the manual clicks). We ignore YOLO!
        data = {"map_path": self.map_image_path, "rooms": self.rooms}
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f)
            print("💾 Memory Saved: User Room Dictionary updated!")
        except Exception as e:
            print(f"Failed to save config: {e}")
            
            #code to take new map
    """
    def upload_new_map(self):
        #Handles user file selection, wipes old memory, and processes the new map.
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Upload Custom Floorplan", "", 
            "Images (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if file_name:
            print(f"Loading new map: {file_name}")
            self.rooms = {} # Wipe old custom pins
            self.map_image_path = file_name 
            
            # Rebuild environment
            self.process_image(file_name)
            self.run_yolo_on_map(file_name)
            self.save_config()
    """        
    #code for calculation on new map 
    
    def upload_new_map(self):
        """Handles user file selection, asks for Width & Length, and mathematically scales the map."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Upload Custom Floorplan", "", 
            "Images (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if file_name:
            # 1. Ask for Width (X-axis)
            width_ft, ok1 = QInputDialog.getDouble(
                self, 'Map Calibration (1/2)', 
                'Enter real-world WIDTH (Left-to-Right) in feet:', 
                50.0, 5.0, 1000.0, 1
            )
            if not ok1: return
            
            # 2. Ask for Length (Y-axis)
            length_ft, ok2 = QInputDialog.getDouble(
                self, 'Map Calibration (2/2)', 
                'Enter real-world LENGTH (Top-to-Bottom) in feet:', 
                40.0, 5.0, 1000.0, 1
            )
            if not ok2: return

            print(f"Loading new map: {width_ft}ft wide by {length_ft}ft long.")
            self.rooms = {} 
            self.map_image_path = file_name 
            
            # 3. Calculate the Uniform Scale
            # We want to fit it inside a 1000x800 box, so we find the strictest boundary
            max_ui_w, max_ui_h = 1000, 800
            pixels_per_foot_w = max_ui_w / width_ft
            pixels_per_foot_h = max_ui_h / length_ft
            
            # Use the smaller scale to ensure the whole map fits on screen!
            self.pixels_per_foot = min(pixels_per_foot_w, pixels_per_foot_h)
            self.pixels_per_inch = self.pixels_per_foot / 12.0
            
            # 4. Lock the new Map Dimensions
            self.map_w = int(width_ft * self.pixels_per_foot)
            self.map_h = int(length_ft * self.pixels_per_foot)
            
            # 5. Sync the 6-inch Circular Prototype
            self.robot_pixel_radius = (self.prototype_diameter_inches / 2.0) * self.pixels_per_inch
            self.wheel_base = self.prototype_diameter_inches * self.pixels_per_inch 
            self.MAX_MOTOR_SPEED = (self.prototype_speed_inches_per_sec * self.pixels_per_inch) / 60.0
            
            print(f"📐 SCALE SET: 1 Foot = {self.pixels_per_foot:.1f} Pixels.")
            print(f"🤖 TWIN SYNC: Robot scaled to {self.robot_pixel_radius * 2:.1f} pixels wide.")

            # Rebuild environment using the new self.map_w and self.map_h
            self.process_image(file_name)
            self.run_yolo_on_map(file_name)
            self.save_config()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    def process_image(self, file_path):
        """
        Converts a visual floorplan into a strict mathematical physics grid using OpenCV.
        Erases text, fuses walls, and builds the Navigation Cost Maps.
        """
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.map_w, self.map_h))
        
        # 1. Binarize: Turn dark pixels into solid walls
        DARKNESS_THRESHOLD = 150  
        _, binary = cv2.threshold(img, DARKNESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Text Eraser: Finds small letter-sized shapes and deletes them from physics
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 65 and h < 65:  
                cv2.drawContours(binary, [contour], -1, 0, -1)

        # ==========================================
        # --- STRUCTURAL INTEGRITY FIX ---
        # ==========================================
        # 3. Window Fuser: Windows are drawn as thin parallel lines. 
        # A 3x3 close operation mathematically bridges the gap, merging them into a thick solid block.
        fuse_kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, fuse_kernel)
                
        # 4. Door Eraser: Door arcs are usually 1 pixel thick. 
        # A 3x3 open operation will destroy the 1px doors but preserve the newly fused thick windows.
        MIN_WALL_THICKNESS = 3
        door_kernel = np.ones((MIN_WALL_THICKNESS, MIN_WALL_THICKNESS), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, door_kernel)
        
        # 5. Fat Shaver: Cleans up the edges
        shave_kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, shave_kernel, iterations=1)

        # 6. Grid Construction
        h_bin, w_bin = binary.shape
        self.grid = np.zeros((h_bin // self.grid_scale, w_bin // self.grid_scale), dtype=int)
        
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                roi = binary[y*self.grid_scale : (y+1)*self.grid_scale, x*self.grid_scale : (x+1)*self.grid_scale]
                
                # Strict Snapping: Block must be 35% dark to become a solid wall
                if np.count_nonzero(roi) > 35:
                    self.grid[y, x] = 1 

        # 7. A* Padding (The Goldilocks Kernel): Widens flat walls but preserves doorway corners
        inflation_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self.path_grid = cv2.dilate(self.grid.astype(np.uint8), inflation_kernel, iterations=1)
        
        # 8. Generate Wall-Fear topographical map
        self.build_navigation_gradient() 
        
        # Create UI background
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h_rgb, w_rgb, ch = img_rgb.shape
        self.map_image = QImage(img_rgb.data, w_rgb, h_rgb, ch * w_rgb, QImage.Format_RGB888).copy()
        
        self.spawn_safely()
        self.update()
        


    # ==========================================
    # --- YOLO AI VISION ---
    # ==========================================
    def run_yolo_on_map(self, file_path):
        """
        Uses CV to paint over text blocks, then runs the YOLO object detection model 
        to find beds, sofas, and TVs.
        """
        if self.yolo_model is None:
            print("WARNING: YOLO model not loaded. Skipping AI detection.")
            return
            
        print("🤖 Scrubbing text with CV, then running YOLO...")
        
        img = cv2.imread(file_path)
        img = cv2.resize(img, (self.map_w, self.map_h)) 
        
        # Digital White-Out: Paint over text so YOLO doesn't hallucinate objects on it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 40 and h < 40:  
                cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)
        
        # Run YOLO with aggressive 0.20 confidence to catch faint sketches
        results = self.yolo_model(img, conf=0.20)
        
        yolo_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id]
                
                yolo_detections.append({
                    'class': cls_name.lower(), 
                    'box': [x1, y1, x2, y2]
                })
                
        print(f"✅ YOLO detected {len(yolo_detections)} objects!")
        
        self.yolo_boxes = yolo_detections
        self.inject_yolo_obstacles(yolo_detections)
        self.save_config()


    def inject_yolo_obstacles(self, yolo_detections):
        """Converts YOLO bounding boxes into solid physics walls in the grid."""
        if self.grid is None: return

        obstacle_classes = [
            'bar_table', 'bath_sink', 'bathtube', 'cat', 
            'coffee_table', 'cooking_table', 'desk', 'dining_table', 'dishwasher', 
            'dog', 'double_bed', 'fridge', 'kitchen_sink', 
            'shower_tube', 'single_bed', 'sofa', 'stilllife', 'stove', 
            'toilet', 'tv_stand', 'wardrobe', 'washingmachine', 'window', 'plant', 'laundary' 
        ]

        for det in yolo_detections:
            detected_class = det['class'].strip().lower().replace(" ", "_") 
            if detected_class in obstacle_classes:
                x1, y1, x2, y2 = det['box']
                
                # ==========================================
                # --- THE FIX: PIXEL-PERFECT MAPPING ---
                # ==========================================
                # Removed the artificial padding. The red blocks will now 
                # perfectly align with the green YOLO bounding boxes!
                gx1 = max(0, int(x1 // self.grid_scale))
                gy1 = max(0, int(y1 // self.grid_scale))
                gx2 = min(self.grid.shape[1], int(math.ceil(x2 / self.grid_scale)))
                gy2 = min(self.grid.shape[0], int(math.ceil(y2 / self.grid_scale)))
                
                self.grid[gy1:gy2, gx1:gx2] = 1 # Solidify
                
        # Re-pad the A* grid with the newly added furniture
        inflation_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self.path_grid = cv2.dilate(self.grid.astype(np.uint8), inflation_kernel, iterations=1)
        self.build_navigation_gradient() 
        
        # Pop-Out: Rescues the wheelchair if a bed spawned on top of it
        self.spawn_safely()
        self.update()

    #custom voice feedback for gaze tracking
    def speak_feedback(self, text):
        """Cross-Platform Voice Engine. Uses a background thread to completely eliminate GUI lag."""
        
        def _speak():
            current_os = platform.system()
            clean_text = text.replace("'", "").replace('"', "")
            
            try:
                if current_os == "Darwin": 
                    # Using subprocess.run is safe here because we are inside a background thread!
                    subprocess.run(['say', clean_text])
                    
                elif current_os == "Windows": 
                    ps_script = f"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak('{clean_text}')"
                    subprocess.run(['powershell', '-Command', ps_script], creationflags=0x08000000)
                    
                else:
                    subprocess.run(['espeak', clean_text])
                    
            except Exception as e:
                print(f"Voice Engine Error: {e}")

        # Instantly spawn the voice in the background without pausing the game!
        threading.Thread(target=_speak, daemon=True).start()


    # =========================================================
    # === ADDED THIS FOR WEBCAM (MOUSE & WEBCAM SEPARATION) ===
    # =========================================================
    def mouseMoveEvent(self, event):
        """Simulates eye tracking by using the MOUSE."""
        # ONLY triggers for the mouse modality!
        if self.active_modality == "Cursor Tracking": 
            self.gaze_x = event.x()
            self.gaze_y = event.y()

    def receive_gaze_coords(self, x, y):
        """Catches real X/Y coordinates from the WEBCAM thread."""
        # ONLY triggers for the webcam modality!
        if self.active_modality == "Custom Map Gaze":
            self.gaze_x = x
            self.gaze_y = y
            self.update() # Force UI refresh for webcam tracking
    # =========================================================


    #code for eye gaze processing
    def process_gaze_dwell(self):
        """Handles the 1.5-second stare-to-click mechanic for routing."""
        # 1. THE AUTOPILOT LOCK: If we are already driving, completely ignore the eyes!
        if getattr(self, 'current_cmd', "") == "AUTOPILOT":
            self.dwell_progress = 0.0
            self.dwell_target = None
            return
        
        # =========================================================
        # === ADDED THIS FOR WEBCAM (GATEKEEPER) ===
        # =========================================================
        # ALLOW BOTH MODALITIES! eye tracking and cursor
        allowed_modes = ["Cursor Tracking", "Custom Map Gaze"]
        if self.active_modality not in allowed_modes: 
            self.dwell_target = None
            self.dwell_progress = 0.0
            return
        # =========================================================

        is_looking_at_target = False
        target_pos = None
        target_name = None

        # 1. Check Custom Room Pins First
        for room_name, pos in self.rooms.items():
            dist = math.hypot(self.gaze_x - pos[0], self.gaze_y - pos[1])
            if dist < 35: 
                is_looking_at_target = True
                target_name = room_name
                target_pos = pos
                break

        # 2. Check YOLO Furniture Second
        if not is_looking_at_target and hasattr(self, 'yolo_boxes'):
            for det in self.yolo_boxes:
                x1, y1, x2, y2 = det['box']
                px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)
                dist = math.hypot(self.gaze_x - px, self.gaze_y - py)
                
                if dist < 40: 
                    is_looking_at_target = True
                    target_name = det['class'].upper()
                    target_pos = (px, py)
                    break

        # ==========================================
        # --- THE FIX: THE DWELL LOCK ---
        # ==========================================
        if is_looking_at_target:
            if self.dwell_target == target_name:
                # Only count up if we haven't hit 100% yet!
                if self.dwell_progress < 1.0: 
                    self.dwell_progress += 1.0 / self.DWELL_THRESHOLD
                    
                # BOOM! 1.5 Seconds reached! Trigger Autopilot exactly ONCE!
                if self.dwell_progress >= 1.0:
                    print(f"👁️ Eye Tracking Engaged: Navigating to {target_name}")
                    
                    # Clean up the name for the Voice Engine (e.g. "DOUBLE_BED" -> "Double Bed")
                    clean_name = target_name.replace("_", " ").title()
                    
                    # 1. Trigger the Audio Feedback!
                    self.speak_feedback(f"Navigating to {clean_name}")
                    
                    # 2. Update the Black Input Feed Box in the Control Panel
                    try:
                        bus.command_received.emit(f"Gaze Locked: {clean_name}")
                    except Exception:
                        pass
                    
                    self.calculate_path(target_pos)
                    # 2. THE RESET FIX: Force the ring back to 0 so it doesn't fire again!
                    self.dwell_progress = 0.0 
                    self.dwell_target = None
            else:
                self.dwell_target = target_name
                self.dwell_progress = 0.0
        else:
            # User looked away, release the lock and reset
            self.dwell_target = None
            self.dwell_progress = 0.0


    # ==========================================
    # --- SPAWNING & RESCUE LOGIC ---
    # ==========================================
    def get_safe_parking_spot(self, target_gx, target_gy):
        """Spirals outward from a solid obstacle to find the nearest safe floor tile."""
        max_search_radius = 20 
        
        if self.path_grid[target_gy, target_gx] == 0:
            return target_gx, target_gy
            
        for radius in range(1, max_search_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check the outer edge of the current square ring
                    if max(abs(dx), abs(dy)) == radius:
                        nx, ny = target_gx + dx, target_gy + dy
                        if 0 <= nx < self.path_grid.shape[1] and 0 <= ny < self.path_grid.shape[0]:
                            if self.path_grid[ny, nx] == 0: 
                                return nx, ny
        return target_gx, target_gy 

    #reset game function for eyetraking
    def reset_game(self):
        """Resets the wheelchair's speed and position when switching maps."""
        self.current_cmd = "STOP"
        self.autopilot_path = []
        if hasattr(self, 'v_l'): self.v_l = 0.0
        if hasattr(self, 'v_r'): self.v_r = 0.0
        self.cruise_active = False
        self.spawn_safely() 
        self.update()



    def spawn_safely(self):
        """Drops the wheelchair into the Living Room, ensuring it doesn't land inside a wall."""
        if self.grid is None: return

        self.v_l, self.v_r = 0.0, 0.0
        self.wheel_anim_left, self.wheel_anim_right = 0, 0
        self.current_cmd = "STOP"
        self.autopilot_path = []
        self.cruise_active = False

        # Target center of map, or LIVING room if available
        target_x, target_y = self.grid.shape[1] * self.grid_scale // 2, self.grid.shape[0] * self.grid_scale // 2
        for room_name, pos in self.rooms.items():
            if "LIVING" in room_name:
                target_x, target_y = pos[0], pos[1]
                break

        # Spiral search up to 50 cells away to escape giant furniture spawns
        gx, gy = int(target_x // self.grid_scale), int(target_y // self.grid_scale)
        if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
            if self.grid[gy, gx] == 1:
                for radius in range(1, 50):
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            ny, nx = gy + dy, gx + dx
                            if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]:
                                if self.grid[ny, nx] == 0:
                                    self.x, self.y = nx * self.grid_scale, ny * self.grid_scale
                                    return 
        self.x, self.y = target_x, target_y


    # ==========================================
    # --- GUI INTERACTION (PINS & COMMANDS) ---
    # ==========================================
    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.btn_edit.setText("Finish Editing")
            self.btn_edit.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
            QMessageBox.information(self, "Edit Mode", "Click anywhere on the floorplan to label a room!")
        else:
            self.btn_edit.setText("Edit Room Labels")
            self.btn_edit.setStyleSheet("background-color: #e67e22; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
            self.save_config() 

    def mousePressEvent(self, event):
        """Handles dropping custom room pins when in Edit Mode."""
        if self.edit_mode and self.map_image:
            room_name, ok = QInputDialog.getText(self, 'Label Room', 'Enter Room Name (e.g., KITCHEN):')
            if ok and room_name:
                clean_name = room_name.strip().upper()
                self.rooms[clean_name] = (event.x(), event.y())
                
                self.save_config() # Instant Memory Sync
                bus.environment_changed.emit("Upload Custom Floorplan") # Force Voice Engine update
                self.update()

    def update_modality(self, modality):
        self.active_modality = modality
        self.update()

    def update_command(self, cmd):
        """Processes manual GUI/Keyboard commands."""
        if cmd == "CRUISE_TOGGLE": 
            self.cruise_active = not self.cruise_active
            self.autopilot_path = [] 
        elif cmd.startswith("NAV_"):
            pass # Voice commands are handled by handle_voice_navigation
        else: 
            self.current_cmd = cmd
            if cmd in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]:
                self.autopilot_path = []


    # ==========================================
    # --- A* PATHFINDING ALGORITHMS ---
    # ==========================================
    def build_navigation_gradient(self):
        """Creates an invisible 'Wall Fear' topographical map to force A* into the center of hallways."""
        if self.grid is None: return
        base_grid = self.grid.astype(np.uint8)

        self.cost_map = np.zeros_like(self.grid, dtype=float)

        # Danger Zone 1: Touching the wall (Massive penalty)
        self.cost_map += cv2.dilate(base_grid, np.ones((5, 5), np.uint8), iterations=1) * 50.0
        # Danger Zone 2: Close to wall (High penalty)
        self.cost_map += cv2.dilate(base_grid, np.ones((9, 9), np.uint8), iterations=1) * 20.0
        # Danger Zone 3: Approaching wall (Mild warning)
        self.cost_map += cv2.dilate(base_grid, np.ones((13, 13), np.uint8), iterations=1) * 5.0
        
        
    def calculate_path(self, target_pos):
        if self.grid is None: return
        
        start_node = (int(self.x // self.grid_scale), int(self.y // self.grid_scale))
        goal_node = (int(target_pos[0] // self.grid_scale), int(target_pos[1] // self.grid_scale))
        active_grid = getattr(self, 'path_grid', self.grid)
        
        # --- RESCUE THE STARTING POSITION ---
        if active_grid[start_node[1], start_node[0]] == 1:
            found_safe = False
            for radius in range(1, 20):
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        ny, nx = start_node[1] + dy, start_node[0] + dx
                        if 0 <= ny < active_grid.shape[0] and 0 <= nx < active_grid.shape[1]:
                            if active_grid[ny, nx] == 0:
                                start_node = (nx, ny)
                                self.x, self.y = nx * self.grid_scale, ny * self.grid_scale 
                                print("⚠️ Wheelchair was stuck in a wall boundary. Nudged to safety!")
                                found_safe = True
                                break
                    if found_safe: break
                if found_safe: break

        # ==========================================
        # --- SMART RESCUE: RAYCAST TO WHEELCHAIR ---
        # ==========================================
        if active_grid[goal_node[1], goal_node[0]] == 1:
            print(f"⚠️ Goal {goal_node} is inside an obstacle. Raycasting to find nearest edge...")
            
            angle = math.atan2(start_node[1] - goal_node[1], start_node[0] - goal_node[0])
            dist = 0
            found_safe = False
            
            while dist < 100:
                check_x = int(goal_node[0] + math.cos(angle) * dist)
                check_y = int(goal_node[1] + math.sin(angle) * dist)
                
                if 0 <= check_y < active_grid.shape[0] and 0 <= check_x < active_grid.shape[1]:
                    if active_grid[check_y, check_x] == 0:
                        goal_node = (check_x, check_y)
                        print(f"✅ Goal rescued to safe edge at {goal_node}!")
                        found_safe = True
                        break
                dist += 1
                
            if not found_safe:
                print("❌ Could not rescue goal. Pathing aborted.")
                self.autopilot_path = []
                return

        def heuristic(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
        
        frontier = []
        heapq.heappush(frontier, (0, start_node))
        came_from = {start_node: None}
        cost_so_far = {start_node: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            if current == goal_node: break
                
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                nx, ny = current[0] + dx, current[1] + dy
                
                if not (0 <= ny < active_grid.shape[0] and 0 <= nx < active_grid.shape[1]): continue
                if active_grid[ny, nx] == 1: continue 
                
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                
                penalty = self.cost_map[ny, nx] if hasattr(self, 'cost_map') else 0.0
                new_cost = cost_so_far[current] + move_cost + penalty
                
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + heuristic(goal_node, (nx, ny))
                    heapq.heappush(frontier, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
                    
        if goal_node in came_from:
            current = goal_node
            path = []
            while current != start_node:
                path.append((current[0] * self.grid_scale + self.grid_scale//2, 
                             current[1] * self.grid_scale + self.grid_scale//2))
                current = came_from[current]
            path.reverse()
            self.autopilot_path = path
            self.current_cmd = "AUTOPILOT"
            print("✅ Centered A* Path successfully calculated and engaged!")
        else:
            print("❌ No valid path found! The destination might be completely blocked by walls.")
            self.autopilot_path = []


    # ==========================================
    # --- LONG RANGE LiDAR SYSTEM ---
    # ==========================================
    def update_lidar_sensors(self):
        """Casts 5 rays up to 120 pixels away to detect upcoming walls."""
        if self.grid is None: return
        self.sensor_angles = [-45, -20, 0, 20, 45]
        self.sensor_distances = []
        max_dist = 120 
        
        for angle_offset in self.sensor_angles:
            ray_angle = math.radians(self.angle + angle_offset)
            dist = 0
            while dist < max_dist:
                check_x = self.x + math.cos(ray_angle) * dist
                check_y = self.y - math.sin(ray_angle) * dist
                gx, gy = int(check_x // self.grid_scale), int(check_y // self.grid_scale)
                
                if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
                    if self.grid[gy, gx] == 1:
                        break # Wall hit!
                else:
                    break # Map edge hit
                dist += 2
            self.sensor_distances.append(dist)

            
    # ==========================================
    # --- DIFFERENTIAL PHYSICS & AUTOPILOT ENGINE ---
    # ==========================================
    def process_physics(self):
        if self.grid is None: return
        
        # Run the Eye Tracking Dwell logic every frame
        self.process_gaze_dwell()
        
        # 1. Update Sensors
        try:
            self.update_lidar_sensors() 
        except Exception:
            pass 
            
        # 2. Safety initialization
        if not hasattr(self, 'v_l'): self.v_l = 0.0
        if not hasattr(self, 'v_r'): self.v_r = 0.0
        if not hasattr(self, 'wheel_anim_left'): self.wheel_anim_left = 0
        if not hasattr(self, 'wheel_anim_right'): self.wheel_anim_right = 0
        if not hasattr(self, 'turn_cooldown'): self.turn_cooldown = 0
            
        if self.turn_cooldown > 0: self.turn_cooldown -= 16  
            
        target_v_l = 0.0
        target_v_r = 0.0
        
        # ==========================================
        # --- FIX 1: LOWER BASE SPEED ---
        # ==========================================
        # Reduced from 2.2 to 1.8. This gives the physics engine enough 
        # time to actually execute turns without drifting into walls!
        MAX_SPEED = 1.8 

        # ==========================================
        # --- BUTTER-SMOOTH AUTOPILOT ---
        # ==========================================
        if self.current_cmd == "AUTOPILOT" and self.autopilot_path:
            
            # ==========================================
            # --- FIX 1: TRUE PURE PURSUIT (NEVER LOOK BACK) ---
            # ==========================================
            # 1. Find the absolute closest waypoint to our current location
            min_dist = float('inf')
            closest_idx = 0
            for i, wp in enumerate(self.autopilot_path):
                dist = math.hypot(self.x - wp[0], self.y - wp[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            # 2. INSTANTLY delete all waypoints behind the wheelchair! 
            self.autopilot_path = self.autopilot_path[closest_idx:]
            
            # 3. Consume the current waypoint if we are close enough
            if min_dist < 25:
                self.autopilot_path.pop(0)
                    
            if not self.autopilot_path:
                self.current_cmd = "STOP" 
            else:
                # Look 3 steps ahead to draw a smooth vector (Ignores grid zig-zags)
                lookahead_idx = min(3, len(self.autopilot_path) - 1)
                target_wp = self.autopilot_path[lookahead_idx]
                
                desired_angle = math.degrees(math.atan2(-(target_wp[1] - self.y), target_wp[0] - self.x))
                
                # ==========================================
                # --- FIX 2: PROPORTIONAL MAGNETIC FORCEFIELDS ---
                # ==========================================
                avoidance_angle = 0.0
                
                if hasattr(self, 'sensor_distances') and len(self.sensor_distances) == 5:
                    r_far = self.sensor_distances[0]   
                    r_mid = self.sensor_distances[1]   
                    center = self.sensor_distances[2]  
                    l_mid = self.sensor_distances[3]    
                    l_far = self.sensor_distances[4]    
                    
                    left_push = 0.0
                    right_push = 0.0
                    
                    # The closer the wall, the harder it pushes!
                    # This naturally cancels out in narrow hallways (stops the violent shaking)
                    if l_mid < 25: right_push += (25 - l_mid) * 1.5
                    if l_far < 20: right_push += (20 - l_far) * 1.5
                    
                    if r_mid < 25: left_push += (25 - r_mid) * 1.5
                    if r_far < 20: left_push += (20 - r_far) * 1.5
                    
                    if center < 25:
                        if r_far > l_far: right_push += (25 - center) * 2.0 # Dodge Right
                        else: left_push += (25 - center) * 2.0 # Dodge Left
                        
                    avoidance_angle = left_push - right_push
                                
                self.target_angle = desired_angle + avoidance_angle
                angle_diff = (self.target_angle - self.angle + 180) % 360 - 180
                
                # Dynamic Braking & Steering Execution
                if abs(angle_diff) > 45.0: 
                    # Pivot for sharp turns 
                    turn_speed = MAX_SPEED * 0.2 # 0.2 to 0.2 speed control, we can change high or low according to our requirement.
                    if angle_diff > 0: target_v_l, target_v_r = -turn_speed, turn_speed
                    else: target_v_l, target_v_r = turn_speed, -turn_speed
                else: 
                    # Straightaways go fast, curves automatically hit the brakes!
                    speed_reduction = (abs(angle_diff) / 45.0) * 0.4
                    forward_speed = MAX_SPEED * (0.85 - speed_reduction) 
                    
                    steering = (angle_diff / 45.0) * (MAX_SPEED * 0.4)
                    
                    target_v_l = forward_speed - steering
                    target_v_r = forward_speed + steering
                    
        # ==========================================
        # --- MANUAL CONTROLS ---
        # ==========================================
        else:
            if self.current_cmd == "BACKWARD": self.cruise_active = False 
                
            if self.cruise_active or self.current_cmd == "FORWARD":
                target_v_l = MAX_SPEED; target_v_r = MAX_SPEED
            elif self.current_cmd == "BACKWARD":
                target_v_l = -MAX_SPEED; target_v_r = -MAX_SPEED

            # ==========================================
            # --- FIX 3: GE3 15-DEGREE MANUAL SNAP ---
            # ==========================================
            # Restored the 1000 cooldown so it locks nicely after a 15 degree turn!
            if self.current_cmd == "LEFT" and self.turn_cooldown <= 0:
                self.target_angle += 15.0; self.turn_cooldown = 1000  
            elif self.current_cmd == "RIGHT" and self.turn_cooldown <= 0:
                self.target_angle -= 15.0; self.turn_cooldown = 1000  

            angle_diff = (self.target_angle - self.angle + 180) % 360 - 180

            if abs(angle_diff) > 0.5:
                turn_speed = MAX_SPEED * 0.4
                if angle_diff > 0: target_v_l = -turn_speed; target_v_r = turn_speed
                else: target_v_l = turn_speed; target_v_r = -turn_speed
            else:
                self.angle = self.target_angle
                
        # ==========================================
        # --- SMOOTH PHYSICS APPLICATION ---
        # ==========================================
        angle_diff_phys = (self.target_angle - self.angle + 180) % 360 - 180
        
        # GE3 acceleration profiles
        if abs(angle_diff_phys) > 0.5: accel = 0.025 
        elif target_v_l == 0 and target_v_r == 0: accel = 0.04 
        else: accel = 0.03 
            
        self.v_l += (target_v_l - self.v_l) * accel
        self.v_r += (target_v_r - self.v_r) * accel
        
        if abs(self.v_l) < 0.005: self.v_l = 0.0
        if abs(self.v_r) < 0.005: self.v_r = 0.0
        
        V = (self.v_r + self.v_l) / 2.0
        omega = (self.v_r - self.v_l) / getattr(self, 'wheel_base', 30.0)
        
        if abs(angle_diff_phys) > 0.5: self.angle += math.degrees(omega)
        self.angle = self.angle % 360
            
        rad = math.radians(self.angle)
        new_x = self.x + math.cos(rad) * V
        new_y = self.y - math.sin(rad) * V
        
        # ==========================================
        # --- GRID COLLISION ---
        # ==========================================
        hit_wall = False
        currently_trapped = False
        
        cgx, cgy = int(self.x // self.grid_scale), int(self.y // self.grid_scale)
        if 0 <= cgy < self.grid.shape[0] and 0 <= cgx < self.grid.shape[1]:
            if self.grid[cgy, cgx] == 1:
                currently_trapped = True


        if not currently_trapped:
            BUMPER = 5 
            # =================================================================
            # ⚠️ HARDWARE REMINDER: DIGITAL TWIN DISCONNECT ⚠️
            # For this UI simulation, BUMPER is set to 25 to match the neon ring.
            # WHEN WRITING THE ARDUINO C++ CODE: This must be reverted to the 
            # physical radius of the prototype (roughly 5 pixels / 3 inches)!
            # =================================================================
            bounds = [(new_x - BUMPER, new_y - BUMPER), (new_x + BUMPER, new_y - BUMPER),
                      (new_x - BUMPER, new_y + BUMPER), (new_x + BUMPER, new_y + BUMPER)]
                      
            for bx, by in bounds:
                gx, gy = int(bx // self.grid_scale), int(by // self.grid_scale)
                if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
                    if self.grid[gy, gx] == 1:
                        hit_wall = True
                        break

        if self.current_cmd == "BACKWARD" and currently_trapped:
            hit_wall = False

        if not hit_wall:
            self.x = new_x
            self.y = new_y
            if abs(self.v_l) > 0.1: self.wheel_anim_left += self.v_l * 1.5
            if abs(self.v_r) > 0.1: self.wheel_anim_right += self.v_r * 1.5
        else:
            self.v_l, self.v_r = 0, 0 
            self.cruise_active = False

        self.update()


    # ==========================================
    # --- RENDER ENGINE (UI OVERLAY) ---
    # ==========================================
    def paintEvent(self, event):
        """Draws the map, physics layers, LiDAR rays, and Wheelchair graphics."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # No Map Fallback
        if self.map_image is None:
            painter.fillRect(self.rect(), QColor(30, 35, 40))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "NO MAP LOADED.\nUse the 'Upload New Floorplan' button.")
            return

        # 1. Base Map
        painter.drawImage(0, 0, self.map_image)
    
        # 2. X-Ray Vision: Physics Grid Overlay
        if self.grid is not None:
            painter.setBrush(QColor(255, 0, 0, 100)) # Semi-transparent red walls
            painter.setPen(Qt.NoPen)
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    if self.grid[y, x] == 1:
                        painter.drawRect(int(x * self.grid_scale), int(y * self.grid_scale), int(self.grid_scale), int(self.grid_scale))
                        
        # 3. Autopilot Path (Dashed Blue Line)
        if self.autopilot_path:
            painter.setPen(QPen(QColor(255, 100, 100, 150), 4, Qt.DashLine))
            current_pos = QPointF(self.x, self.y)
            for wp in self.autopilot_path:
                next_pos = QPointF(wp[0], wp[1])
                painter.drawLine(current_pos, next_pos)
                current_pos = next_pos
            painter.setBrush(QColor(255, 100, 100))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(current_pos, 8, 8) # Final target dot

        # 4. Custom Room Pins (Red Dots)
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        for name, pos in self.rooms.items():
            painter.setBrush(QColor(231, 76, 60))
            painter.setPen(QPen(QColor(192, 57, 43), 2))
            painter.drawEllipse(QPointF(pos[0], pos[1]), 8, 8)
            
            rect = painter.fontMetrics().boundingRect(name)
            rect.moveTo(int(pos[0] + 12), int(pos[1] - 12))
            rect.adjust(-4, -2, 4, 2)
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(Qt.NoPen)
            painter.drawRect(rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(pos[0] + 12), int(pos[1]), name)

        # 5. Glowing Green LiDAR Rays
        if hasattr(self, 'sensor_distances') and self.sensor_distances:
            painter.setPen(QPen(QColor(0, 255, 100, 200), 2)) 
            for i, dist in enumerate(self.sensor_distances):
                ray_angle = math.radians(self.angle + self.sensor_angles[i])
                end_x = self.x + math.cos(ray_angle) * dist
                end_y = self.y - math.sin(ray_angle) * dist
                painter.drawLine(QPointF(self.x, self.y), QPointF(end_x, end_y))
        #wheelchair square tank design
        """
        # 6. Draw Wheelchair Chassis
        painter.save()
        painter.translate(self.x, self.y)
        painter.rotate(-self.angle)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 90)) # Shadow
        painter.drawRoundedRect(QRectF(-22, -22, 55, 45), 5, 5)

        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(QColor(10, 10, 10), 2))
        painter.drawRoundedRect(QRectF(-20, -24, 35, 8), 3, 3) 
        painter.drawRoundedRect(QRectF(-20, 16, 35, 8), 3, 3)  
        """
        # New wheelchair design for digital twin 
        """
        # 5. Draw the Circular Digital Twin Prototype
        painter.save()
        painter.translate(self.x, self.y)
        painter.rotate(-self.angle)
        
        # Get the strict mathematical radius (fallback to 5.0 if not set yet)
        safe_radius = max(3.0, getattr(self, 'robot_pixel_radius', 5.0))
        
        # --- 1. VISUAL RADAR HALO (So your human eyes can find it!) ---
        painter.setBrush(QColor(0, 255, 255, 40)) # Light transparent cyan
        painter.setPen(QPen(QColor(0, 255, 255, 200), 1, Qt.DashLine))
        painter.drawEllipse(QPointF(0, 0), 25, 25) # Large 50-pixel glowing ring
        
        # --- 2. STRICT PHYSICS CHASSIS (The actual 6-inch robot) ---
        painter.setBrush(QColor(255, 50, 50, 255)) # Bright Red so it stands out
        painter.setPen(QPen(QColor(255, 255, 255), 2)) # Solid white edge
        painter.drawEllipse(QPointF(0, 0), safe_radius, safe_radius)
        
        # --- 3. DIRECTIONAL NOSE (Shows where the front is pointing) ---
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        # Draw the line slightly past the actual body so you can see it turning
        painter.drawLine(QPointF(0, 0), QPointF(safe_radius + 10, 0))
        
        painter.restore()
        
        # Animated Tank Treads
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
        painter.setBrush(QColor(0, 255, 0, 200)) # Front Nose
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(24, -4), QPointF(24, 4), QPointF(32, 0)]))
        painter.restore()


"""


    # =========================================================
        # 5. DRAW WHEELCHAIR CHASSIS & DIGITAL TWIN OVERLAY
        # =========================================================
        painter.save()
        
        # Lock both the animation and the hit-box to the exact same coordinates!
        painter.translate(self.x, self.y)
        painter.rotate(-self.angle)
        
        # --- A. DRAW THE ANIMATED WHEELCHAIR ---
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 90)) # Shadow
        painter.drawRoundedRect(QRectF(-22, -22, 55, 45), 5, 5)

        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(QColor(10, 10, 10), 2))
        painter.drawRoundedRect(QRectF(-20, -24, 35, 8), 3, 3) 
        painter.drawRoundedRect(QRectF(-20, 16, 35, 8), 3, 3)  
        
        # Animated Tank Treads
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
        
        # --- B. DRAW THE DIGITAL TWIN HITBOX OVERLAY ---
        # Safe fallback in case the map hasn't fully loaded the radius yet
        safe_radius = max(3.0, getattr(self, 'robot_pixel_radius', 5.0))
        
        # Visual Radar Halo
        painter.setBrush(QColor(0, 255, 255, 40)) 
        painter.setPen(QPen(QColor(0, 255, 255, 200), 1, Qt.DashLine))
        painter.drawEllipse(QPointF(0, 0), 25, 25) 
        
        # Strict Physics Chassis (The Red Arduino Box)
        painter.setBrush(QColor(255, 50, 50, 255)) 
        painter.setPen(QPen(QColor(255, 255, 255), 2)) 
        painter.drawEllipse(QPointF(0, 0), safe_radius, safe_radius)
        
        # Directional Nose (Green Line)
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        painter.drawLine(QPointF(0, 0), QPointF(safe_radius + 15, 0))
        
        painter.restore()




    
        # 7. Draw YOLO AI Vision (Neon Green Boxes)
        if hasattr(self, 'yolo_boxes') and self.yolo_boxes:
            for det in self.yolo_boxes:
                x1, y1, x2, y2 = det['box']
                cls_name = det['class'].upper()
                
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 50, 0, 200)) 
                
                text_width = painter.fontMetrics().width(cls_name) + 10
                label_rect = QRectF(x1, y1 - 18, text_width, 18)
                painter.drawRect(label_rect)
                
                painter.setPen(QColor(0, 255, 0)) 
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(int(x1) + 5, int(y1) - 5, cls_name)

        # =========================================================
        # === ADDED THIS FOR WEBCAM (ALLOW BOTH TO DRAW) ===
        # =========================================================
        if self.active_modality in ["Cursor Tracking", "Custom Map Gaze"]:
            
            # 1. Draw the Base Gaze Dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 255, 100)) # Transparent Cyan
            painter.drawEllipse(QPointF(self.gaze_x, self.gaze_y), 20, 20)
            
            painter.setBrush(QColor(0, 255, 255, 255)) # Solid Core
            painter.drawEllipse(QPointF(self.gaze_x, self.gaze_y), 4, 4)
            
            # 2. Draw the Dwell Loading Ring
            if self.dwell_target and self.dwell_progress > 0:
                painter.setPen(QPen(QColor(0, 255, 100), 4, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                
                # Draw a pie-slice arc that fills up
                span_angle = int(self.dwell_progress * 360 * 16) 
                painter.drawArc(
                    int(self.gaze_x - 30), int(self.gaze_y - 30), 
                    60, 60, 
                    90 * 16, -span_angle
                )
        # =========================================================

    # ==========================================
    # --- VOICE NAVIGATION ROUTER ---
    # ==========================================
    def handle_voice_navigation(self, cmd):
        """Catches NAV_ commands and finds a safe parking spot for Autopilot."""
        if cmd.startswith("NAV_"):
            target_name = cmd.replace("NAV_", "").lower()
            print(f"⚙️ Game Engine processing route to: {target_name}...")

            # 1. Prioritize Custom User Pins
            for custom_room, pos in self.rooms.items():
                if target_name in custom_room.lower() or custom_room.lower() in target_name:
                    print(f"📍 Found custom room pin for: {custom_room}")
                    self.calculate_path(pos) 
                    self.update()
                    return

            # 2. Fallback to YOLO Extracted Locations
            if hasattr(self, 'yolo_boxes'):
                for det in self.yolo_boxes:
                    if det['class'].lower() == target_name:
                        x1, y1, x2, y2 = det['box']
                        px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        
                        raw_gx, raw_gy = int(px // self.grid_scale), int(py // self.grid_scale)
                        
                        # Find nearest safe floor tile outside the furniture hit-box
                        safe_gx, safe_gy = self.get_safe_parking_spot(raw_gx, raw_gy)

                        safe_pixel_x = safe_gx * self.grid_scale
                        safe_pixel_y = safe_gy * self.grid_scale

                        print(f"📍 Found YOLO target for: {target_name}")
                        self.calculate_path((safe_pixel_x, safe_pixel_y))
                        self.update()
                        return
            
            print(f"❌ Error: Could not find map coordinates for {target_name}!")
            
            """
        # =========================================================
        # --- NEW: LIVE PERFORMANCE HUD OVERLAY ---
        # =========================================================
        import time
        current_time = time.time()
        
        if not hasattr(self, 'last_render_time'):
            self.last_render_time = current_time
            self.fps_filter = 60.0
            
        delta = current_time - self.last_render_time
        if delta > 0:
            inst_fps = 1.0 / delta
            self.fps_filter = (0.9 * self.fps_filter) + (0.1 * inst_fps) 
            
        self.last_render_time = current_time
        cycle_ms = delta * 1000

        painter.setBrush(QColor(0, 0, 0, 200))
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.drawRoundedRect(20, 20, 210, 65, 8, 8)

        painter.setPen(QColor(0, 255, 255))
        painter.setFont(QFont("Courier", 14, QFont.Bold))
        
        if self.fps_filter < 30:
            painter.setPen(QColor(255, 50, 50))
            
        painter.drawText(35, 45, f"RENDER FPS: {int(self.fps_filter)}")
        
        painter.setPen(QColor(0, 255, 255))
        painter.drawText(35, 70, f"CYCLE TIME: {int(cycle_ms)} ms")
"""
    # ==========================================
    # --- KEYBOARD CONTROLS (OVERRIDE) ---
    # ==========================================
    def keyPressEvent(self, event):
        """Allows manual driving, instantly overriding Voice/Autopilot."""
        
        # Shield: Ignore OS key-spamming 
        if event.isAutoRepeat():
            return

        # Emergency Override: Touching WASD instantly cancels the AI path!
        if self.current_cmd == "AUTOPILOT":
            print("Manual Override Triggered! Canceling AI Autopilot.")
            self.autopilot_path = [] 

        if event.key() == Qt.Key_W or event.key() == Qt.Key_Up:
            self.current_cmd = "FORWARD"
        elif event.key() == Qt.Key_S or event.key() == Qt.Key_Down:
            self.current_cmd = "BACKWARD"
        elif event.key() == Qt.Key_A or event.key() == Qt.Key_Left:
            self.current_cmd = "LEFT"
        elif event.key() == Qt.Key_D or event.key() == Qt.Key_Right:
            self.current_cmd = "RIGHT"
        elif event.key() == Qt.Key_Space:
            self.current_cmd = "STOP"

    def keyReleaseEvent(self, event):
        """Stops the wheelchair when you let go of the key."""
        if event.isAutoRepeat():
            return
            
        if event.key() in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            self.current_cmd = "STOP"
            
    