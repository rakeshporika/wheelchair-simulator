import os
import warnings

# --- 1. SILENCE C++ AND MEDIAPIPE LOGS ---
# 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES' # Critical macOS fix for OpenCV

# --- 2. SILENCE PROTOBUF PYTHON WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import sys
import csv     
import json                                           
from datetime import datetime             
import pyqtgraph as pg
import cv2             
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QStackedWidget, 
                             QPushButton, QFrame, QInputDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import helper

# Import our custom modules
from signals import bus
from styles import MODERN_STYLE
from game_engine import MazeWidget
from head_tracking import VisionWorker
from eye_tracking import EyeTrackingWorker
from voice_control import VoiceWorker
from custom_voice_control import CustomVoiceWorker
from custom_map_gaze import CustomGazeWorker #<-- for eye gaze in custom maps
from eeg_control import EEGWorker
from game_engine_2 import ArenaWidget 
from game_engine_3 import HomeArenaWidget 
from game_engine_4 import DynamicHomeWidget 


# --- Helper Component ---
class ControlStatusItem(QWidget):
    def __init__(self, title, subtitle, color="#2ecc71"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        
        icon = QLabel()
        icon.setFixedSize(24, 24)
        icon.setStyleSheet(f"background-color: {color}; border-radius: 12px;")
        
        text_layout = QVBoxLayout()
        t = QLabel(title)
        t.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.sub = QLabel(subtitle)
        self.sub.setStyleSheet("color: gray; font-size: 10px;")
        text_layout.addWidget(t)
        text_layout.addWidget(self.sub)
        
        layout.addWidget(icon)
        layout.addLayout(text_layout)
        layout.addStretch()

    def update_text(self, text):
        self.sub.setText(text)

# --- Main Window ---
class ModernDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # ==========================================
        # --- RESEARCH PARADIGM: LOGIN & LOGGING ---
        # ==========================================
        name, ok = QInputDialog.getText(self, 'Participant Login', 'Enter Participant Name:')
        self.participant_name = name if (ok and name) else "Unknown_Participant"
        
        self.current_modality = "Keyboard Controls" 
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_folder = "CSVs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder) 
            
        self.log_filename = os.path.join(log_folder, f"{self.participant_name}_SessionLog_{timestamp}.csv")
        
        with open(self.log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Participant Name", "Active Modality", "Action / Command"])
            
        # Connect signals to loggers
        bus.data_logged.connect(self.log_detailed_event) 
        bus.modality_changed.connect(self.log_modality)
        bus.environment_changed.connect(self.log_environment)
        bus.environment_changed.connect(self.switch_voice_engine)
        bus.collision_occurred.connect(self.log_collision)
        self.last_crash_time = 0 
        
        self.log_detailed_event(self.current_modality, "System Initialized & Session Started")
        
        self.setStyleSheet(MODERN_STYLE)
        self.setWindowTitle("Assistive Tech Control Center")
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ==========================================
        # 1. LEFT SIDE: The Navigation Grid (80%)
        # ==========================================
        self.game_stack = QStackedWidget()
        
        self.practice_map = MazeWidget()
        self.arena_map = ArenaWidget()
        self.home_map = HomeArenaWidget() 
        self.custom_map = DynamicHomeWidget() 
        
        self.game_stack.addWidget(self.practice_map) 
        self.game_stack.addWidget(self.arena_map)    
        self.game_stack.addWidget(self.home_map)     
        self.game_stack.addWidget(self.custom_map)   
        
        main_layout.addWidget(self.game_stack, stretch=8)

        # ==========================================
        # 2. RIGHT SIDE: Status, Camera & Controls (20%)
        # ==========================================
        right_col = QFrame()
        right_col.setObjectName("Card")
        right_col.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_col)
        
        # --- EEG Display ---
        right_layout.addWidget(QLabel("EEG Signal", objectName="Title"))
        self.eeg_plot = pg.PlotWidget()
        self.eeg_plot.setBackground('white')
        self.eeg_plot.setFixedHeight(100)
        self.eeg_plot.setYRange(-40, 40) 
        self.eeg_plot.hideAxis('bottom') 
        self.eeg_plot.hideAxis('left')
        self.eeg_line = self.eeg_plot.plot(pen=pg.mkPen(color='#9b59b6', width=2))
        bus.eeg_data_updated.connect(self.eeg_line.setData)
        right_layout.addWidget(self.eeg_plot)
        right_layout.addSpacing(10)

        # --- Camera Feed ---
        right_layout.addWidget(QLabel("Live Input Feed", objectName="Title"))
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setStyleSheet("background-color: black; border-radius: 8px;")
        self.camera_feed.setFixedSize(320, 180) 
        right_layout.addWidget(self.camera_feed)
        bus.frame_updated.connect(lambda img: self.camera_feed.setPixmap(QPixmap.fromImage(img)))
        right_layout.addSpacing(10)

        # --- Controls ---
        right_layout.addWidget(QLabel("Control Panel", objectName="Title"))

        right_layout.addWidget(QLabel("Active Modality:"))
        self.modality_dropdown = QComboBox()
        self.modality_dropdown.addItems(["Keyboard Controls", "Head Tracking", "Eye Tracking", "Voice Commands", "Cursor Tracking", "Custom Map Gaze", "EEG Brainwaves"])
        self.modality_dropdown.setStyleSheet("padding: 5px; border-radius: 4px; border: 1px solid #bdc3c7;")
        self.modality_dropdown.currentIndexChanged.connect(self.switch_modality)
        right_layout.addWidget(self.modality_dropdown)
        right_layout.addSpacing(10)

        right_layout.addWidget(QLabel("Environment:"))
        self.map_combo = QComboBox()
        self.map_combo.addItems(["Practice Grid", "Obstacle Arena", "Full House Layout", "Upload Custom Floorplan"])
        self.map_combo.setStyleSheet("padding: 5px; border-radius: 4px; border: 1px solid #bdc3c7;")
        self.map_combo.currentIndexChanged.connect(self.change_map)
        right_layout.addWidget(self.map_combo)
        right_layout.addSpacing(10)
        
        self.current_env = self.map_combo.currentText()
        
        # Status Indicators
        self.input_status = ControlStatusItem("Current Input", "Idle", "#3498db")
        right_layout.addWidget(self.input_status)
        right_layout.addWidget(ControlStatusItem("EEG System", "Standby", "#2ecc71"))
        
        bus.command_received.connect(lambda cmd: self.input_status.update_text(f"Command: {cmd}"))
        right_layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset Game", objectName="Reset")
        reset_btn.clicked.connect(self.reset_active_game)
        btn_layout.addWidget(reset_btn)
        
        exit_btn = QPushButton("Exit", objectName="Exit")
        exit_btn.clicked.connect(self.close)
        btn_layout.addWidget(exit_btn)
        
        right_layout.addLayout(btn_layout)
        main_layout.addWidget(right_col, stretch=2)

        # --- Initialize the Default Worker ---
        self.active_thread = None
        self.switch_modality(0) 
        self.setFocusPolicy(Qt.StrongFocus)

    # ==========================================
    # --- CORE UI METHODS ---
    # ==========================================
    def change_map(self, index):
        """Switches the visible game engine and resets it."""
        self.game_stack.setCurrentIndex(index)
        bus.environment_changed.emit(self.map_combo.currentText())
        self.reset_active_game()

    def reset_active_game(self):
        """Resets whichever map is currently visible on screen."""
        current_widget = self.game_stack.currentWidget()
        current_widget.reset_game()

    def switch_modality(self, index):
        """Safely stops the current thread without freezing the GUI."""
        modality_name = self.modality_dropdown.currentText()
        bus.modality_changed.emit(modality_name)
        
        if self.active_thread is not None:
            self.active_thread.running = False
            if not self.active_thread.wait(500):
                self.active_thread.terminate()
                self.active_thread.wait()
            self.active_thread = None 
                
        # ==========================================
        # --- THE FIX: INDEX-PROOF ROUTING ---
        # ==========================================
        if "Keyboard" in modality_name or index == 0:
            self.update_keyboard_feed("STOP")
            
        elif modality_name == "Head Tracking" or modality_name == "Computer Vision":
            self.active_thread = VisionWorker()
            
        elif modality_name == "Eye Tracking":
            self.active_thread = EyeTrackingWorker()
            
        elif "Voice" in modality_name:
            current_map = getattr(self, 'current_env', "")
            if current_map == "Upload Custom Floorplan": 
                self.active_thread = CustomVoiceWorker()
            else:
                self.active_thread = VoiceWorker()
                
        elif modality_name == "EEG System":
            self.active_thread = EEGWorker()   
            
        elif modality_name == "Cursor Tracking":
            self.active_thread = None

        # =========================================================
        # === ADDED THIS FOR WEBCAM (WIRING X/Y AND VIDEO) ===
        # =========================================================
        elif modality_name == "Custom Map Gaze":
            self.active_thread = CustomGazeWorker()
            
            # 1. Send coordinates to the CURRENTLY VISIBLE map (not a hardcoded 'current_widget')
            active_map = self.game_stack.currentWidget()
            if hasattr(active_map, 'receive_gaze_coords'):
                self.active_thread.gaze_coords.connect(active_map.receive_gaze_coords)
                
            # 2. Send the video frame to your preview box using the corrected method below!
            self.active_thread.frame_processed.connect(self.update_gaze_video_feed)
            
            # 3. Force the EEG graph to hide itself since it sometimes turns on accidentally
            if hasattr(self, 'eeg_plot'):
                self.eeg_plot.hide()
        # =========================================================
                
        if self.active_thread is not None:
            self.active_thread.start()

        self.setFocus()
    
    
    def switch_voice_engine(self, env_name):
        """Reboots the voice thread if the user changes maps while the mic is live."""
        self.current_env = env_name
        if self.modality_dropdown.currentIndex() == 3:
            self.switch_modality(3)

    def closeEvent(self, event):
        """Ensures threads and cameras shut down cleanly when closing the app."""
        if self.active_thread is not None:
            self.active_thread.running = False
            if not self.active_thread.wait(500):
                self.active_thread.terminate()
        event.accept()

    # ==========================================
    # --- DETECTION & JSON MAPPING LOGIC ---
    # ==========================================
    def run_detection(self):
        """Scans the image, generates the A* grid, and saves room coordinates."""
        # Safety check to prevent crashes if the model hasn't loaded yet
        if not hasattr(self, 'model') or not hasattr(self, 'current_frame'):
            print("Warning: YOLO model or frame not initialized yet.")
            return

        # 1. YOLO scans the image
        results = self.model.predict(self.current_frame, conf=0.25)
        self.detections = results[0].boxes
        
        # 2. Create the grid for A*
        self.nav_grid = helper.create_navigation_grid(self.current_frame.shape, self.detections)
        
        # 3. Extract targets and filter out invalid/outside objects
        detected_rooms = {}
        
        for box in self.detections:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id].lower()
            
            # Filter out random artifacts, alphabets, or non-house items
            invalid_items = ['alphabet', 'text', 'size', 'letter']
            if any(invalid in class_name for invalid in invalid_items):
                continue
                
            # Restrict saving to only these valid internal obstacle/target classes
            valid_targets = ['wall', 'bed', 'table', 'chair', 'dining table', 'bathroom sink', 'toilet', 'sofa', 'kitchen']
            if not any(valid in class_name for valid in valid_targets):
                 continue
            
            # Get the center coordinates of the object for the wheelchair
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            detected_rooms[class_name] = [center_x, center_y]
            
        # Format and save the dictionary for the Voice Worker
        map_data = {"rooms": detected_rooms}
        with open("custom_map_config.json", "w") as f:
            json.dump(map_data, f, indent=4)
            
        print(f"Saved new map data: {list(detected_rooms.keys())}")
        
        # 4. Trigger the custom voice worker to reload (Check if it's the active thread)
        if hasattr(self, 'active_thread') and isinstance(self.active_thread, CustomVoiceWorker):
            self.active_thread.reload_map_vocabulary()
            
        self.update()

    # ==========================================
    # --- DATA LOGGING FUNCTIONS ---
    # ==========================================
    def log_detailed_event(self, modality, action_details):
        """Logs cleanly to the CSV file AND prints live to the terminal."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S") 
        
        print(f"[{time_str}] {modality}: {action_details}")
        
        try:
            with open(self.log_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([date_str, time_str, self.participant_name, modality, action_details])
        except Exception as e:
            print(f"Logging Error: {e}")

    def log_modality(self, modality_name):
        """Logs whenever the user switches the input control method."""
        self.current_modality = modality_name
        self.log_detailed_event("System", f"Switched Modality to {modality_name}")

    def log_environment(self, env):
        self.log_detailed_event(self.current_modality, f"System: Switched Environment to {env}")

    def log_collision(self, obstacle_name):
        """Logs collisions to the CSV, but limits it to one log per 2 seconds to prevent spam."""
        import time
        current_time = time.time()
        
        if current_time - self.last_crash_time > 2.0:
            self.last_crash_time = current_time
            self.log_detailed_event(self.current_modality, f"CRITICAL: Collision with {obstacle_name}")
            
            
    # =========================================================
    # === ADDED THIS FOR WEBCAM (VIDEO PREVIEW BOX) ===
    # =========================================================
    def update_gaze_video_feed(self, qt_image):
        """Catches the video frame from Custom Map Gaze and displays it."""
        from PyQt5.QtGui import QPixmap
        
        # Properly fixed to use 'self.camera_feed' from your __init__ function!
        if hasattr(self, 'camera_feed'): 
            self.camera_feed.setPixmap(QPixmap.fromImage(qt_image))
            self.camera_feed.setScaledContents(True)
    # =========================================================


    # ==========================================
    # --- KEYBOARD VISUAL FEEDBACK ENGINE ---
    # ==========================================
    def update_keyboard_feed(self, cmd):
        """Draws a live WASD graphic to the camera feed."""
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "KEYBOARD ACTIVE", (75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        c_fwd = (0, 255, 0) if cmd == "FORWARD" else (70, 70, 70)
        c_back = (0, 255, 0) if cmd == "BACKWARD" else (70, 70, 70)
        c_left = (0, 255, 0) if cmd == "LEFT" else (70, 70, 70)
        c_right = (0, 255, 0) if cmd == "RIGHT" else (70, 70, 70)

        cv2.rectangle(frame, (140, 40), (180, 80), c_fwd, -1)   
        cv2.rectangle(frame, (90, 90), (130, 130), c_left, -1)  
        cv2.rectangle(frame, (140, 90), (180, 130), c_back, -1) 
        cv2.rectangle(frame, (190, 90), (230, 130), c_right, -1)

        cv2.putText(frame, "W", (150, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, "A", (102, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, "S", (152, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, "D", (202, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        bus.frame_updated.emit(qt_img)    

    # ==========================================
    # --- DEFAULT KEYBOARD CONTROLS ---
    # ==========================================
    def keyPressEvent(self, event):
        """Native PyQt key listener. Only triggers if Keyboard mode is active."""
        if event.isAutoRepeat() or self.current_modality != "Keyboard Controls":
            return
            
        cmd = ""
        if event.key() in (Qt.Key_W, Qt.Key_Up): cmd = "FORWARD"
        elif event.key() in (Qt.Key_S, Qt.Key_Down): cmd = "BACKWARD"
        elif event.key() in (Qt.Key_A, Qt.Key_Left): cmd = "LEFT"
        elif event.key() in (Qt.Key_D, Qt.Key_Right): cmd = "RIGHT"
        elif event.key() == Qt.Key_Space: cmd = "CRUISE_TOGGLE"

        if cmd:
            bus.command_received.emit(cmd)
            self.update_keyboard_feed(cmd)
            if cmd != "CRUISE_TOGGLE":
                self.log_detailed_event("Keyboard Controls", f"Action: {cmd} | Input: Keystroke")

    def keyReleaseEvent(self, event):
        """Stops the wheelchair the exact millisecond the user lets go of the key."""
        if event.isAutoRepeat() or self.current_modality != "Keyboard Controls":
            return

        if event.key() in (Qt.Key_W, Qt.Key_Up, Qt.Key_S, Qt.Key_Down, Qt.Key_A, Qt.Key_Left, Qt.Key_D, Qt.Key_Right):
            bus.command_received.emit("STOP")
            self.update_keyboard_feed("STOP")
            self.log_detailed_event("Keyboard Controls", "Action: STOP | Input: Key Released")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernDashboard()
    window.showMaximized()
    sys.exit(app.exec_())