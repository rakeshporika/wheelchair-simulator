import json
import numpy as np
import cv2
import pyaudio
import time  
import threading  
import pyttsx3    
import queue   
import math   
import platform      # <--- NEW: Automatically detects Mac vs Windows
import subprocess    # <--- NEW: Runs Apple's native voice
from vosk import Model, KaldiRecognizer
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QImage
from signals import bus

class VoiceWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.last_heard = ""
        self.wheelchair_active = False  
        
        self.last_action_time = 0  
        self.current_env = "Full House Layout" 
        bus.environment_changed.connect(self.update_env)
        
        # ==========================================
        # --- SMART CROSS-PLATFORM TTS ENGINE ---
        # ==========================================
        self.os_type = platform.system()
        
        # We ONLY start the pyttsx3 background thread if you are on Windows!
        if self.os_type == "Windows":
            self.tts_queue = queue.Queue()
            threading.Thread(target=self._tts_loop, daemon=True).start()

    def _tts_loop(self):
        """This loop only runs on Windows."""
        engine = pyttsx3.init()
        engine.setProperty('rate', 175) 
        while True:
            text = self.tts_queue.get()
            if text is None: break 
            engine.say(text)
            engine.runAndWait()

    def update_env(self, env_name):
        self.current_env = env_name

    def stop(self):
        self.running = False
        if self.os_type == "Windows":
            self.tts_queue.put(None) 
        self.wait()

    def speak(self, text):
        """Instantly uses the correct, zero-latency voice for your exact operating system."""
        if self.os_type == "Darwin":  # If you are currently on your Mac
            # FIXED: Dropped the speed from 190 down to 150 for a much calmer pace
            subprocess.Popen(["say", "-r", "150", text])
        elif self.os_type == "Windows":  # If you move to your Windows PC
            self.tts_queue.put(text)
        
    def update_ui(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "VOICE COMMANDS", (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.wheelchair_active:
            cv2.rectangle(frame, (0,0), (320, 180), (0, 255, 0), 4) 
            cv2.putText(frame, "SYSTEM: ACTIVE", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Say 'Rio stop my wheelchair'", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.rectangle(frame, (0,0), (320, 180), (0, 0, 255), 4) 
            cv2.putText(frame, "SYSTEM: LOCKED", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Say 'Rio start my wheelchair'", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if self.last_heard:
            cv2.putText(frame, f"Heard: '{self.last_heard}'", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
        bus.frame_updated.emit(qt_img)

    def process_text(self, text):
        if not text: return
        words = text.split()
        
        # If less than 1.5 seconds have passed since the last action, ignore this text!
        current_time = time.time()
        if current_time - self.last_action_time < 1.5:
            return

        cmd_triggered = False

        if all(w in words for w in ["rio", "start", "wheelchair"]):
            if not self.wheelchair_active: 
                self.wheelchair_active = True
                self.last_heard = "Rio start my wheelchair"
                bus.command_received.emit("STOP") 
                self.speak("Wheelchair activated.") 
                self.update_ui()
                self.last_action_time = current_time 
            return
            
        if all(w in words for w in ["rio", "lock", "wheelchair"]):
            if self.wheelchair_active: 
                self.wheelchair_active = False
                self.last_heard = "Rio lock my wheelchair"
                bus.command_received.emit("LOCK")
                self.speak("Wheelchair locked.") 
                self.update_ui()
                self.last_action_time = current_time 
            return

        if self.wheelchair_active and "rio" in words:
            
            # --- ONLY ALLOW NAVIGATION IN ARENA 3 ---
            if self.current_env == "Full House Layout":
                if "kitchen" in words: 
                    bus.command_received.emit("NAV_KITCHEN")
                    self.speak("Navigating to Kitchen")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_KITCHEN | Feedback: 'Navigating to Kitchen'")
                    cmd_triggered = True
                elif "living" in words: 
                    bus.command_received.emit("NAV_LIVING_ROOM")
                    self.speak("Navigating to Living Room")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_LIVING_ROOM | Feedback: 'Navigating to Living Room'")
                    cmd_triggered = True
                elif "garage" in words: 
                    bus.command_received.emit("NAV_GARAGE")
                    self.speak("Navigating to Garage")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_GARAGE | Feedback: 'Navigating to Garage'")
                    cmd_triggered = True
                elif "bathroom" in words: 
                    bus.command_received.emit("NAV_BATH")
                    self.speak("Navigating to Bathroom")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_BATH | Feedback: 'Navigating to Bathroom'")
                    cmd_triggered = True
                elif "garden" in words: 
                    bus.command_received.emit("NAV_GARDEN")
                    self.speak("Navigating to Garden")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_GARDEN | Feedback: 'Navigating to Garden'")
                    cmd_triggered = True
                elif "one" in words: 
                    bus.command_received.emit("NAV_BEDROOM1")
                    self.speak("Navigating to Bedroom 1")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_BEDROOM1 | Feedback: 'Navigating to Bedroom 1'")
                    cmd_triggered = True
                elif "two" in words: 
                    bus.command_received.emit("NAV_BEDROOM2")
                    self.speak("Navigating to Bedroom 2")
                    bus.data_logged.emit("Voice Commands", "Action: NAV_BEDROOM2 | Feedback: 'Navigating to Bedroom 2'")
                    cmd_triggered = True
            
            # --- MANUAL DRIVING WORKS IN ALL ARENAS ---
            if not cmd_triggered:
                if any(w in words for w in ["left"]): 
                    bus.command_received.emit("LEFT")
                    bus.data_logged.emit("Voice Commands", "Action: LEFT | Feedback: None")
                    cmd_triggered = True
                elif any(w in words for w in ["right"]): 
                    bus.command_received.emit("RIGHT")
                    bus.data_logged.emit("Voice Commands", "Action: RIGHT | Feedback: None")
                    cmd_triggered = True
                elif any(w in words for w in ["go","forward", "straight"]): 
                    bus.command_received.emit("FORWARD")
                    bus.data_logged.emit("Voice Commands", "Action: FORWARD | Feedback: None")
                    cmd_triggered = True
                elif any(w in words for w in ["back", "backward", "reverse"]): 
                    bus.command_received.emit("BACKWARD")
                    bus.data_logged.emit("Voice Commands", "Action: BACKWARD | Feedback: None")
                    cmd_triggered = True
                elif any(w in words for w in ["stop"]): 
                    bus.command_received.emit("STOP")
                    bus.data_logged.emit("Voice Commands", "Action: STOP | Feedback: None")
                    cmd_triggered = True
            if cmd_triggered:
                self.last_heard = text
                self.last_action_time = current_time 
                self.update_ui()

    def run(self):
        self.wheelchair_active = False
        bus.command_received.emit("LOCK")
        self.update_ui()
        
        try:
            ai_model = Model("model")
            allowed_words = '["rio", "go", "start", "my", "wheelchair", "left", "right", "forward", "straight", "back", "backward", "reverse", "stop", "lock", "kitchen", "living", "garage", "bathroom", "garden", "one", "two", "[unk]"]'
            recognizer = KaldiRecognizer(ai_model, 16000, allowed_words)
            
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1000)
            stream.start_stream()
            
            while self.running:
                data = stream.read(800, exception_on_overflow=False)
                if len(data) == 0: continue
                
                # ==========================================
                # --- NOISE GATE COMPLETELY REMOVED ---
                # ==========================================
                # Instantly reads partial results without checking volume
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    self.process_text(result.get("text", ""))
                else:
                    partial_result = json.loads(recognizer.PartialResult())
                    self.process_text(partial_result.get("partial", ""))
                        
        except Exception as e:
            print(f"Vosk Error: {e}")
            self.last_heard = "Error: Check model folder!"
            self.update_ui()
            
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()