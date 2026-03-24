import time
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtCore import QThread
from pylsl import StreamInlet, resolve_stream  # <--- NEW: For live hardware streaming
from signals import bus

# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class BrainwaveDecoder(nn.Module):
    """
    Feed-Forward Neural Network to classify live 16-channel g.tec EEG data.
    """
    def __init__(self):
        super(BrainwaveDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)   # Classes: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. LIVE HARDWARE STREAMING ENGINE
# ==========================================
class EEGWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.commands = ["STOP", "FORWARD", "LEFT", "RIGHT"]
        
        # Initialize the AI Model
        self.model = BrainwaveDecoder()
        self.model.eval() 
        
        self.graph_data = [0] * 100 
        self.current_cmd = "STOP"
        self.inlet = None

    def stop(self):
        self.running = False
        self.wait()

    def connect_to_headset(self):
        """Looks for the g.tec LSL stream on the local network."""
        print("Looking for an EEG stream...")
        # Resolves any active LSL stream marked as 'EEG'
        streams = resolve_stream('type', 'EEG')
        
        if len(streams) > 0:
            # Connect to the first found stream (your g.tec headset)
            self.inlet = StreamInlet(streams[0])
            print(f"Successfully connected to hardware stream: {streams[0].name()}")
            return True
        else:
            print("No EEG stream found. Make sure g.tec software is broadcasting via LSL!")
            return False

    def run(self):
        # 1. Try to connect to the physical headset
        connected = False
        while self.running and not connected:
            connected = self.connect_to_headset()
            if not connected:
                self.msleep(2000) # Wait 2 seconds and try again
                
        if not self.running:
            return

        # 2. Main Live Hardware Loop
        while self.running:
            # Pull the absolute latest microvolt sample from the g.tec headset
            # sample is a list of floats (the 16 channel voltages)
            sample, timestamp = self.inlet.pull_sample()
            
            if sample:
                # We only need the first 16 channels (sometimes LSL adds trigger channels at the end)
                eeg_channels = sample[:16]
                
                # Convert the live voltages to a PyTorch Tensor
                input_tensor = torch.FloatTensor(eeg_channels).unsqueeze(0) 
                
                # Neural Network Inference
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    predicted_class = torch.argmax(logits, dim=1).item()
                
                cmd = self.commands[predicted_class]
                
                # Broadcast and log the command
                if cmd != self.current_cmd:
                    self.current_cmd = cmd
                    bus.command_received.emit(cmd)
                    if cmd != "STOP":
                        bus.data_logged.emit("g.tec EEG", f"Network Classified Intent: {cmd}")

                # Update the Dashboard Graph using Channel 1 (Frontal/Motor cortex usually looks best)
                self.graph_data.append(eeg_channels[0]) 
                self.graph_data.pop(0)
                bus.eeg_data_updated.emit(self.graph_data)
                
            # Run fast to keep up with the hardware's sampling rate (usually 250Hz+)
            self.msleep(10)