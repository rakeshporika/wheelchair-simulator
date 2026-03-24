import json
import os
import numpy as np
import cv2
import pyaudio
import time  
import threading  
import queue   
import platform      
import subprocess    
from vosk import Model, KaldiRecognizer
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QImage
from signals import bus

class CustomVoiceWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.last_heard = ""
        self.wheelchair_active = False  
        self.last_action_time = 0  
        
        self.os_type = platform.system()
        self.tts_queue = queue.Queue()
        self.dynamic_rooms = []
        
        try:
            self.ai_model = Model("model") 
        except Exception as e:
            print(f"❌ Failed to load Vosk Model. Error: {e}")
            self.ai_model = None
            
        self.recognizer = None 
        
        threading.Thread(target=self._tts_loop, daemon=True).start()

    def _tts_loop(self):
        # Fallback TTS for Windows (or if needed)
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 220) 
        while True:
            text = self.tts_queue.get()
            if text is None: break 
            engine.say(text)
            engine.runAndWait()

    def stop(self):
        self.running = False
        self.tts_queue.put(None) 
        self.wait()

    def speak(self, text):
        if self.os_type == "Darwin":  
            subprocess.Popen(["say", "-r", "150", text])
        else:
            self.tts_queue.put(text)
    """
    def update_ui(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "CUSTOM MAP VOICE", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        if self.wheelchair_active:
            cv2.rectangle(frame, (0,0), (320, 180), (0, 255, 0), 4) 
            cv2.putText(frame, "SYSTEM: ACTIVE", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (0,0), (320, 180), (0, 0, 255), 4) 
            cv2.putText(frame, "SYSTEM: LOCKED", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.last_heard:
            cv2.putText(frame, f"Heard: '{self.last_heard}'", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
        bus.frame_updated.emit(qt_img)
    """   
    #code from original voice control
    def update_ui(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "CUSTOM VOICE COMMANDS", (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        
        #exesting code
        
    def reload_map_vocabulary(self):
        """Rebuilds the AI Dictionary with ultra-high accuracy grammar."""
        if self.ai_model is None: return
        try:
            self.dynamic_rooms = []
            if os.path.exists("custom_map_config.json"):
                with open("custom_map_config.json", 'r') as f:
                    data = json.load(f)
                    self.dynamic_rooms = list(data.get("rooms", {}).keys())
            
            # Base mandatory vocabulary (added common room words to ensure they are recognized)
            vocab = ["rio", "go", "start", "my", "wheelchair", "left", "right", "forward", "straight", "back", "backward", "reverse", "stop", "lock", "navigate", "to", "the", "[unk]", "kitchen", "bedroom", "bathroom", "living", "room", "hall"]
            
            # Inject new YOLO targets and custom labels
            for room in self.dynamic_rooms:
                clean_room_name = room.replace("_", " ")
                vocab.extend(clean_room_name.lower().split())
                
            allowed_words = json.dumps(list(set(vocab))) 
            
            # Restored strict grammar for high accuracy!
            self.recognizer = KaldiRecognizer(self.ai_model, 16000, allowed_words)
            print(f"✅ Voice System Updated! Listening for targets: {self.dynamic_rooms}")
            
        except Exception as e:
            print(f"Error reloading map dictionary: {e}")
            # Safe fallback so it doesn't crash
            self.recognizer = KaldiRecognizer(self.ai_model, 16000)

    def process_text(self, text):
        """Processes the final sentence."""
        if not text or text == "[unk]": return
        
        print(f"🗣️ AI OFFICIALLY HEARD: '{text}'")
        self.last_heard = text
        self.update_ui()
        
        words = text.split()
        
        current_time = time.time()
        if current_time - self.last_action_time < 1.0: return

        cmd_triggered = False

        # --- INSTANT SAFETY CONTROLS ---
        if "rio" in words and "start" in words:
            if not self.wheelchair_active: 
                self.wheelchair_active = True
                self.last_heard = "Rio start my wheelchair"
                bus.command_received.emit("STOP") 
                self.update_ui()
                self.speak("Custom map system activated.") 
                self.last_action_time = current_time 
            return
            
        if "rio" in words and "lock" in words:
            if self.wheelchair_active: 
                self.wheelchair_active = False
                self.last_heard = "Rio lock my wheelchair"
                bus.command_received.emit("LOCK")
                self.update_ui()
                self.speak("System locked.") 
                self.last_action_time = current_time 
            return

        # --- DYNAMIC NAVIGATION ---
        if self.wheelchair_active and "rio" in words:
            
            # Basic commands
            if any(w in words for w in ["stop"]): 
                bus.command_received.emit("STOP")
                cmd_triggered = True
            elif any(w in words for w in ["forward", "straight"]): 
                bus.command_received.emit("FORWARD")
                cmd_triggered = True
            elif any(w in words for w in ["left"]): 
                bus.command_received.emit("LEFT")
                cmd_triggered = True
            elif any(w in words for w in ["right"]): 
                bus.command_received.emit("RIGHT")
                cmd_triggered = True
            elif any(w in words for w in ["back", "backward", "reverse"]): 
                bus.command_received.emit("BACKWARD")
                cmd_triggered = True
                
            # Check for specific room names detected by YOLO or Custom Pins
            if not cmd_triggered:
                clean_words = [w for w in words if w not in ["rio", "go", "to", "the", "navigate"]]
                
                for room in self.dynamic_rooms:
                    room_clean = room.replace("_", " ")
                    room_words = room_clean.lower().split()
                    
                    if all(rw in clean_words for rw in room_words):
                        target = f"NAV_{room.upper()}"
                        bus.command_received.emit(target)
                        self.speak(f"Navigating to {room_clean}")
                        cmd_triggered = True
                        break

            if cmd_triggered:
                self.last_action_time = current_time 
    """
    def run(self):
        self.wheelchair_active = False
        bus.command_received.emit("LOCK")
        self.update_ui()
        
        try:
            self.reload_map_vocabulary()
            
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1000)
            stream.start_stream()
            
            
            
            print("🎙️ Microphone is LIVE.")
            
            while self.running:
                data = stream.read(800, exception_on_overflow=False)
                if len(data) == 0: continue
                
                if self.recognizer and self.recognizer.AcceptWaveform(data):
                    # Only process the fully completed sentence
                    result = json.loads(self.recognizer.Result())
                    self.process_text(result.get("text", ""))
                elif self.recognizer:
                    # Just update the screen with mid-sentence guesses, DO NOT process logic!
                    partial_result = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "")
                    if partial_text and partial_text != "[unk]":
                        self.last_heard = partial_text + "..."
                        self.update_ui()
                        
        except Exception as e:
            print(f"Vosk Error: {e}")
            self.last_heard = "Error: Check audio/model!"
            self.update_ui()
            
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()
    """
    #new code to fix repeated feedback in voice and eyetracking.
    def run(self):
        self.wheelchair_active = False
        bus.command_received.emit("LOCK")
        self.update_ui()
        
        try:
            self.reload_map_vocabulary()
            
            p = pyaudio.PyAudio()
            # Mac-friendly settings: Increased frames_per_buffer to 4096 to stop AUHAL crashes
            stream = p.open(format=pyaudio.paInt16, 
                            channels=1, 
                            rate=16000, 
                            input=True, 
                            frames_per_buffer=4096)
            stream.start_stream()
            
            print("🎙️ Microphone is LIVE.")
            
            while self.running:
                # Read 4000 bytes at a time to match the new 4096 buffer
                data = stream.read(4000, exception_on_overflow=False)
                if len(data) == 0: continue
                
                if self.recognizer and self.recognizer.AcceptWaveform(data):
                    # Only process the fully completed sentence
                    result = json.loads(self.recognizer.Result())
                    self.process_text(result.get("text", ""))
                elif self.recognizer:
                    # Just update the screen with mid-sentence guesses, DO NOT process logic!
                    partial_result = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "")
                    if partial_text and partial_text != "[unk]":
                        self.last_heard = partial_text + "..."
                        self.update_ui()
                        
        except Exception as e:
            print(f"Vosk Error: {e}")
            self.last_heard = "Error: Check audio/model!"
            self.update_ui()
            
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()