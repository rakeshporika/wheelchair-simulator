import numpy as np
import cv2
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage
from signals import bus

class EEGWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.data_buffer = [0] * 100  # 100 points for the rolling graph
        self.time_step = 0

    def stop(self):
        """Safely stops the thread."""
        self.running = False
        self.wait()

    def run(self):
        while self.running:
            # 1. GENERATE SIMULATED BRAINWAVES
            noise = np.random.normal(0, 3)
            base_wave = np.sin(self.time_step * 0.5) * 5 # Relaxed Alpha wave
            
            # Simulate BCI Paradigms (cycle through states every few seconds)
            cycle = self.time_step % 100
            
            if cycle < 40:
                # State 1: Relaxed (Low amplitude) -> STOP
                wave_val = base_wave + noise
                current_cmd = "STOP"
                state_text = "Baseline (Alpha)"
            
            elif 40 <= cycle < 70:
                # State 2: High Concentration (Fast, high amplitude Beta waves) -> FORWARD
                wave_val = np.sin(self.time_step * 2.0) * 15 + noise
                current_cmd = "FORWARD"
                state_text = "Focus (Beta)"
                
            else:
                # State 3: Jaw Clench Artifact (Massive spike) -> RIGHT
                wave_val = 30 + noise
                current_cmd = "RIGHT"
                state_text = "Muscle Artifact"

            # 2. UPDATE GRAPH BUFFER
            self.data_buffer.pop(0)
            self.data_buffer.append(float(wave_val))
            
            # Emit data to the graph and command to the wheelchair
            bus.eeg_data_updated.emit(self.data_buffer)
            bus.command_received.emit(current_cmd)

            # 3. UPDATE UI CAMERA SCREEN
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, "EEG BCI ACTIVE", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 89, 182), 2)
            cv2.putText(frame, f"Wave: {state_text}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            status_color = (0, 255, 0) if current_cmd == "FORWARD" else (0, 165, 255) if current_cmd != "STOP" else (100, 100, 100)
            cv2.putText(frame, f"CMD: {current_cmd}", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            h, w, ch = frame.shape
            qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
            bus.frame_updated.emit(qt_img)

            self.time_step += 1
            self.msleep(50) # Update 20 times per second for a smooth graph