"""
src/tracking  –  Member C
Exports the three public classes used by Member D's FastAPI server.
"""

from .tracker        import BagTracker
from .face_recognizer import FaceRecognizer
from .alert_engine   import AlertEngine, Alert

__all__ = ["BagTracker", "FaceRecognizer", "AlertEngine", "Alert"]
