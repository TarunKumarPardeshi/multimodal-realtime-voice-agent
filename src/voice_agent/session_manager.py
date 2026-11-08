"""
Streaming Session Manager
Tracks stateful audio buffers, visual context history, and tool execution state.
"""

from typing import List, Dict, Any, Optional
import time

class StreamingSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.conversation_turns: List[Dict[str, Any]] = []
        self.current_audio_buffer = bytearray()
        self.last_keyframe: Optional[Dict[str, Any]] = None
        self.agent_is_speaking = False
        self.total_tokens_consumed = 0

    def append_audio_chunk(self, chunk: bytes) -> None:
        self.current_audio_buffer.extend(chunk)

    def flush_audio_buffer(self) -> bytes:
        data = bytes(self.current_audio_buffer)
        self.current_audio_buffer.clear()
        return data

    def record_turn(self, role: str, content: str, visual_context: Optional[str] = None) -> None:
        self.conversation_turns.append({
            "role": role,
            "content": content,
            "visual_context": visual_context,
            "timestamp": time.time()
        })
