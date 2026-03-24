# ♿ Multi-Modal Assistive Robotics Simulator (Digital Twin)

An advanced, multi-modal control dashboard and physics simulator designed for next-generation assistive wheelchairs and robotics.

This platform serves as a **Digital Twin** for a physical 6-inch circular Arduino prototype. It allows users to navigate complex, dynamically generated environments using everything from standard keyboards to real-time dual-eye tracking and voice commands, ensuring that simulated movements mathematically translate to real-world hardware execution.

---

## ✨ Core Features

### 🤖 The Digital Twin Engine

* **Real-World Scaling:** Upload any custom 2D floorplan and input the physical dimensions (feet/inches). The engine mathematically scales the map to a 1000x800 window without distortion.
* **1:1 Prototype Sync:** The virtual robot's wheelbase, turning speed, and physical collision hit-boxes are dynamically generated based on the physical measurements of the Arduino prototype.
* **Visual vs. Physical Hitboxes:** Features a glowing 25-pixel "Radar Halo" for UI visibility, while maintaining a strict, hidden physical collision bumper to ensure the hardware can safely pass through real-world doorways.

### 🎮 Multi-Modal Input System

Seamlessly switch between control paradigms on the fly:

* **👁️ Dual-Eye Gaze Tracking:** Uses a dedicated background thread and Google MediaPipe FaceMesh to track both irises. Applies exponential smoothing for highly stable cursor control.
* **🗣️ Voice Commands:** Offline, on-device NLP processing using Vosk for navigating to specific rooms or objects, complete with cross-platform text-to-speech (TTS) audio feedback.
* **⌨️ Keyboard Controls:** Standard WASD manual override with custom acceleration/deceleration physics.
* **🧠 EEG Brainwave Integration:** Live signal plotting via `pyqtgraph` for BCI (Brain-Computer Interface) research.

### 🧠 AI Autopilot & Pathfinding

* **YOLOv8 Object Detection:** The engine automatically scrubs text from uploaded floorplans and identifies furniture (beds, sofas, tables) as navigable targets and physical obstacles.
* **A* Navigation Grid:** Converts raw images into strict mathematical physics grids with topographical "wall-fear" cost maps to keep the robot centered in hallways.
* **Smart Rescue Raycasting:** Prevents the robot from spawning inside walls or driving into unreachable coordinates.

### 🎯 UI & Analytics

* **Gaze-Dwell UI:** A floating, responsive "Burger Menu" that allows users to select navigation targets simply by staring at a bounding box for 1.5 seconds.
* **Live Performance HUD:** Real-time tracking of Render FPS and Physics Cycle Time (ms) to monitor system latency.
* **Clinical Data Logging:** Automatically generates timestamped CSV logs of all participant interactions and modality switches.

---

## 📂 Project Architecture

```
main.py
```

The central PyQt5 application, UI router, and data logger.

```
game_engine_4.py
```

The core Digital Twin physics engine, A* pathfinding, YOLO bounding-box logic, UI rendering, and hardware scaling math.

```
custom_map_gaze.py
```

Background `QThread` dedicated exclusively to processing MediaPipe FaceMesh webcam feeds for jitter-free dual-eye tracking.

```
custom_voice_control.py
```

Dedicated Vosk NLP listener that strips wake-words and routes targets to the event bus.

```
signals.py
```

Custom PyQt event bus for cross-thread communication.

```
helper.py
```

Utility functions for loading YOLO models and formatting navigation grids.

---

## 🛠️ Prerequisites & Installation

This project requires **Python 3.8+**.
It utilizes heavy computer vision and audio processing libraries.

### 1️⃣ Audio Drivers (Mac Users Only)

If you are on macOS, you must install the underlying C++ audio headers before installing the Python libraries:

```bash
brew install portaudio
```

### 2️⃣ Install Python Dependencies

Activate your virtual environment, then install the required packages:

```bash
pip install PyQt5 pyqtgraph numpy opencv-python torch ultralytics mediapipe vosk PyAudio
```

**Note:**
Ensure you have your **YOLOv8 weights file** `best.pt` and your **Vosk language model** placed inside your root project directory before running the application.

---

## 🚀 How to Run & Calibrate

### 1️⃣ Launch the Main Application

Open your terminal and run:

```bash
python main.py
```

### 2️⃣ Select Environment

Click **Upload Custom Floorplan** and choose a `.jpg` or `.png` image of a house layout.

### 3️⃣ Calibrate the Twin

When prompted, enter the real-world **Width** and **Length** of the floorplan in feet.

The engine will automatically:

* calculate the `pixels_per_inch` ratio
* scale the virtual robot
* match the physical prototype

### 4️⃣ Set Custom Pins

Click **Edit Room Labels** to drop custom navigation targets
(e.g., `"KITCHEN"`) directly onto the map.

### 5️⃣ Navigate

Select your input modality from the right-side control panel:

* Voice
* Custom Map Gaze
* Keyboard

Then engage the autopilot 🚀
