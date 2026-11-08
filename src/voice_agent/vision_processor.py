"""
Vision and Frame Ingestion Processor
Extracts, downsamples, and prepares camera frames for visual reasoning alongside speech.
"""

import base64
import hashlib
from typing import Dict, Any, Optional

class VisionProcessor:
    def __init__(self, max_fps: int = 2):
        self.max_fps = max_fps
        self.last_frame_hash: Optional[str] = None
        self.cached_frame_analysis: Optional[str] = None

    def process_frame(self, frame_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """
        Ingests a raw video frame, hashes it for deduplication, and prepares multimodal payload.
        """
        frame_hash = hashlib.md5(frame_bytes).hexdigest()
        is_keyframe = (frame_hash != self.last_frame_hash)
        self.last_frame_hash = frame_hash

        b64_data = base64.b64encode(frame_bytes).decode("utf-8")

        return {
            "is_keyframe": is_keyframe,
            "mime_type": mime_type,
            "byte_size": len(frame_bytes),
            "base64_encoded": b64_data[:32] + "...",  # Preview for logging
            "full_base64": b64_data
        }

    def describe_scene_heuristic(self, frame_info: Dict[str, Any]) -> str:
        """
        Heuristic / fallback visual description generator when external vision LLM is offline.
        """
        size = frame_info.get("byte_size", 0)
        return f"Live camera feed active: {size} bytes, clean image feed detected."
