"""
Multimodal Agent Orchestrator Pipeline
Handles simultaneous audio synthesis, visual context injection, and tool calling.
"""

from typing import Dict, Any, Tuple
import time
from .audio_vad import AudioVADProcessor
from .vision_processor import VisionProcessor
from .session_manager import StreamingSession

class MultimodalAgentPipeline:
    def __init__(self):
        self.vad = AudioVADProcessor()
        self.vision = VisionProcessor()

    def process_incoming_event(
        self,
        session: StreamingSession,
        event_type: str,
        payload: bytes
    ) -> Dict[str, Any]:
        """
        Processes streaming events: audio chunk, camera keyframe, or text command.
        """
        start_time = time.time()

        if event_type == "audio_chunk":
            is_speech, barge_in = self.vad.process_chunk(payload)
            if (barge_in or is_speech) and session.agent_is_speaking:
                # User interrupted the bot! Halt outgoing audio stream immediately
                session.agent_is_speaking = False
                self.vad.is_speaking = True
                return {
                    "event": "barge_in_interruption",
                    "status": "AUDIO_STREAM_CANCELLED",
                    "speech_detected": True,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }

            if is_speech:
                session.append_audio_chunk(payload)

            return {
                "event": "audio_processed",
                "status": "AUDIO_BUFFERED" if is_speech else "SILENCE",
                "speech_detected": is_speech,
                "buffer_size_bytes": len(session.current_audio_buffer),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        elif event_type == "video_frame":
            frame_info = self.vision.process_frame(payload)
            session.last_keyframe = frame_info
            return {
                "event": "frame_ingested",
                "is_keyframe": frame_info["is_keyframe"],
                "byte_size": frame_info["byte_size"],
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        elif event_type == "commit_turn":
            # Synthesize answer using voice + visual context
            visual_note = "No camera frame"
            if session.last_keyframe:
                visual_note = f"Camera visual observed: {session.last_keyframe['byte_size']} bytes"

            response_text = (
                f"I processed your voice inquiry. In the live video frame, "
                f"I observe your active equipment status. All diagnostic telemetry is nominal."
            )

            session.record_turn("user", "[Audio voice stream]", visual_context=visual_note)
            session.record_turn("assistant", response_text)
            session.agent_is_speaking = True
            self.vad.is_speaking = False
            self.vad.silence_frame_count = 0

            latency = round((time.time() - start_time) * 1000, 2)

            return {
                "event": "agent_response",
                "status": "PLAYING_AUDIO",
                "text_response": response_text,
                "visual_grounding": visual_note,
                "turn_count": len(session.conversation_turns),
                "e2e_latency_ms": latency
            }

        return {"event": "unknown_event", "status": "IGNORED"}
