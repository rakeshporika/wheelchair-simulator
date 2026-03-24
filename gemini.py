import sys
import random
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# ================= SIGNALS & WORKERS ================= #

class CommunicationBus(QObject):
    command_received = pyqtSignal(str)
    frame_updated = pyqtSignal(QImage) # New signal for the camera feed

bus = CommunicationBus()

class VisionWorker(QThread):
    def run(self):
        import mediapipe as mp
        import cv2
        
        mp_face = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils # Used to draw the green face mask
        face_mesh = mp_face.FaceMesh(refine_landmarks=False)
        cap = cv2.VideoCapture(0)
        
        current_cmd = "STOP"
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face mesh on the frame
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                
                nose = results.multi_face_landmarks[0].landmark[1]
                
                # Head tilt logic
                if nose.x < 0.40: 
                    current_cmd = "LEFT"
                elif nose.x > 0.60: 
                    current_cmd = "RIGHT"
                elif nose.y < 0.40: 
                    current_cmd = "FORWARD"
                elif nose.y > 0.60:
                    current_cmd = "BACKWARD"
                else: 
                    current_cmd = "STOP"
                
                bus.command_received.emit(current_cmd)
            else:
                current_cmd = "NO FACE"
                bus.command_received.emit("STOP")

            # --- OVERLAY TEXT AND SEND FRAME TO UI ---
            # Draw the command text inside the video frame
            cv2.putText(frame, f"CMD: {current_cmd}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Convert OpenCV image (BGR) to PyQt image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale the image to fit the UI panel
            scaled_img = qt_img.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            bus.frame_updated.emit(scaled_img)
            
            self.msleep(30)

# ================= STYLING (QSS) ================= #

MODERN_STYLE = """
    QMainWindow { background-color: #f4f7f9; }
    QWidget#Card { background-color: white; border-radius: 12px; border: 1px solid #e0e6ed; }
    QLabel { color: #2c3e50; font-family: 'Arial'; font-size: 13px; }
    QLabel#Title { font-weight: bold; font-size: 16px; color: #34495e; }
    QPushButton { font-family: 'Arial'; font-weight: bold; }
    QPushButton#Reset { background-color: #3498db; color: white; border-radius: 6px; padding: 8px; }
    QPushButton#Exit { background-color: #e74c3c; color: white; border-radius: 6px; padding: 8px; }
    QPushButton#Submit { background-color: #2ecc71; color: white; border-radius: 6px; padding: 8px; }
"""

# ================= MAZE & GAME ENGINE ================= #

class MazeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        
        # 0: Floor, 1: Wall, 2: Coin, 3: Goal
        self.maze = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 2, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 2, 1, 0, 0, 0, 0, 2, 0, 1],
            [1, 2, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 0, 0, 0, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        
        self.start_pos = [1, 1]
        self.player_pos = list(self.start_pos)
        self.current_cmd = "STOP"
        self.goal_reached = False
        
        bus.command_received.connect(self.update_command)
        
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_movement)
        self.move_timer.start(400) 
        
    def update_command(self, cmd):
        self.current_cmd = cmd

    def reset_game(self):
        self.player_pos = list(self.start_pos)
        self.goal_reached = False
        self.current_cmd = "STOP"
        self.update()

    def process_movement(self):
        if self.goal_reached or self.current_cmd == "STOP" or self.current_cmd == "NO FACE": 
            return

        r, c = self.player_pos
        if self.current_cmd == "LEFT": c -= 1
        elif self.current_cmd == "RIGHT": c += 1
        elif self.current_cmd == "FORWARD": r -= 1
        elif self.current_cmd == "BACKWARD": r += 1

        if 0 <= r < len(self.maze) and 0 <= c < len(self.maze[0]):
            if self.maze[r][c] != 1:
                self.player_pos = [r, c]
                if self.maze[r][c] == 3:
                    self.goal_reached = True
        
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rows, cols = len(self.maze), len(self.maze[0])
        w, h = self.width(), self.height()
        cell_size = int(min(w / cols, h / rows))
        
        offset_x = (w - (cols * cell_size)) // 2
        offset_y = (h - (rows * cell_size)) // 2
        
        for r, row in enumerate(self.maze):
            for c, tile in enumerate(row):
                rect = QRect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
                
                if tile == 1:
                    painter.setBrush(QColor("#bdc3c7"))
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(rect, 4, 4)
                elif tile == 2:
                    painter.setBrush(QColor("#f1c40f"))
                    painter.setPen(Qt.NoPen)
                    coin_rect = QRect(rect.center().x() - cell_size//6, rect.center().y() - cell_size//6, cell_size//3, cell_size//3)
                    painter.drawEllipse(coin_rect)
                elif tile == 3:
                    painter.setBrush(QColor("#e74c3c"))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(rect)
                elif tile == 0:
                    painter.setBrush(QColor("#e8f8f5"))
                    painter.setPen(QPen(QColor("#d1f2eb"), 1))
                    painter.drawRect(rect)

        p_size = int(cell_size * 0.6)
        px = offset_x + self.player_pos[1] * cell_size + (cell_size - p_size) // 2
        py = offset_y + self.player_pos[0] * cell_size + (cell_size - p_size) // 2
        
        painter.setBrush(QColor("#3498db"))
        painter.setPen(QPen(QColor("#2980b9"), 2))
        painter.drawEllipse(px, py, p_size, p_size)
        
        if self.goal_reached:
            painter.setBrush(QColor(46, 204, 113, 200))
            painter.drawRect(self.rect())
            painter.setPen(Qt.white)
            font = QFont("Arial", 36, QFont.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "🎉 GOAL REACHED! 🎉\nTask Completed")

# ================= UI PANELS ================= #

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
        text_layout.setSpacing(0)
        
        layout.addWidget(icon)
        layout.addLayout(text_layout)
        layout.addStretch()

    def update_text(self, text):
        self.sub.setText(text)

# ================= MAIN WINDOW ================= #

class ModernDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(MODERN_STYLE)
        self.setWindowTitle("Assive Tech Control Center")
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- LEFT COLUMN: EEG, GAZE & CAMERA ---
        left_col = QFrame()
        left_col.setObjectName("Card")
        left_col.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_col)
        
        left_layout.addWidget(QLabel("EEG Signal", objectName="Title"))
        
        self.eeg_plot = pg.PlotWidget()
        self.eeg_plot.setBackground('white')
        self.eeg_plot.setFixedHeight(120) # Made slightly smaller to fit camera
        left_layout.addWidget(self.eeg_plot)
        
        gaze_box = QFrame()
        gaze_box.setStyleSheet("background: #ebf5fb; border: 1px dashed #3498db; border-radius: 8px;")
        gaze_box.setFixedHeight(100) # Made slightly smaller
        left_layout.addWidget(QLabel("Eye Gaze Map", objectName="Title"))
        left_layout.addWidget(gaze_box)
        
        # NEW: Camera Feed Label
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Head Tracking Camera", objectName="Title"))
        
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setStyleSheet("background-color: black; border-radius: 8px;")
        self.camera_feed.setFixedSize(320, 240) # 4:3 Aspect Ratio for Webcam
        left_layout.addWidget(self.camera_feed)
        
        bus.frame_updated.connect(self.update_camera_frame) # Connect signal to UI
        
        left_layout.addStretch()

        # --- CENTER COLUMN: THE GAME ---
        center_col = QVBoxLayout()
        self.game = MazeWidget()
        center_col.addWidget(self.game, stretch=10)

        # --- RIGHT COLUMN: STATUS ---
        right_col = QFrame()
        right_col.setObjectName("Card")
        right_col.setMaximumWidth(280)
        right_layout = QVBoxLayout(right_col)
        
        right_layout.addWidget(QLabel("Control Status", objectName="Title"))
        
        self.head_status = ControlStatusItem("Head Tracking", "Idle", "#9b59b6")
        right_layout.addWidget(self.head_status)
        right_layout.addWidget(ControlStatusItem("EEG", "Standby", "#2ecc71"))
        right_layout.addWidget(ControlStatusItem("Voice", "Muted", "#95a5a6"))
        
        bus.command_received.connect(lambda cmd: self.head_status.update_text(f"Command: {cmd}"))
        
        right_layout.addSpacing(20)
        right_layout.addWidget(QLabel("Physiological Monitoring", objectName="Title"))
        right_layout.addWidget(QLabel("❤️ 82 bpm"))
        
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset", objectName="Reset")
        reset_btn.clicked.connect(self.game.reset_game)
        
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(QPushButton("Exit", objectName="Exit"))
        btn_layout.addWidget(QPushButton("Submit", objectName="Submit"))
        right_layout.addLayout(btn_layout)
        right_layout.addStretch()

        main_layout.addWidget(left_col, stretch=1)
        main_layout.addLayout(center_col, stretch=3)
        main_layout.addWidget(right_col, stretch=1)

        # Start Camera Background Thread
        self.vision_thread = VisionWorker()
        self.vision_thread.start()

    def update_camera_frame(self, image):
        """Updates the QLabel with the new webcam frame"""
        self.camera_feed.setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernDashboard()
    window.showMaximized()
    sys.exit(app.exec_())