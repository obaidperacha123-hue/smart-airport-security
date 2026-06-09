"""src/tracking  –  Member C
Public exports for the tracking + face recognition + alert subsystem.
Consumed by Member D's FastAPI server (server.py).
"""
 
from .tracker         import BagTracker
from .face_recognizer import FaceRecognizer
from .alert_engine    import AlertEngine, Alert
 
__all__ = ["BagTracker", "FaceRecognizer", "AlertEngine", "Alert"]
 
