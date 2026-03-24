import functools
from ultralytics import YOLO
import settings

# functools.lru_cache ensures the heavy YOLO model is only loaded into your Mac's RAM once!
@functools.lru_cache(maxsize=1)
def load_yolo_model():
    print(f"🤖 Loading Custom YOLOv8 Model from {settings.DETECTION_MODEL}...")
    try:
        model = YOLO(str(settings.DETECTION_MODEL))
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return None