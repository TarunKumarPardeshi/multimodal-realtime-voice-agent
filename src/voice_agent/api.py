"""
FastAPI Server with WebSocket Streaming for Full-Duplex Voice & Vision
Author: Tarun Pardeshi <tarun.tanny1998@gmail.com>
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from typing import Dict, Any
import uuid

from .session_manager import StreamingSession
from .agent_pipeline import MultimodalAgentPipeline

app = FastAPI(
    title="Real-Time Multimodal Voice & Vision Agent",
    description="Sub-300ms Full-Duplex WebRTC / WebSocket Streaming AI Agent",
    version="1.0.0"
)

pipeline = MultimodalAgentPipeline()
active_sessions: Dict[str, StreamingSession] = {}

class SessionInitResponse(BaseModel):
    session_id: str
    status: str
    sample_rate: int
    max_fps: int

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "service": "multimodal-realtime-voice-agent",
        "active_sessions_count": len(active_sessions),
        "sub_300ms_mode": True
    }

@app.post("/api/v1/session/start", response_model=SessionInitResponse)
def start_session():
    sid = str(uuid.uuid4())
    session = StreamingSession(session_id=sid)
    active_sessions[sid] = session
    return SessionInitResponse(
        session_id=sid,
        status="ACTIVE",
        sample_rate=16000,
        max_fps=2
    )

@app.websocket("/ws/stream/{session_id}")
async def websocket_stream_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = active_sessions.get(session_id)
    if not session:
        session = StreamingSession(session_id=session_id)
        active_sessions[session_id] = session

    try:
        while True:
            # Receive either binary audio/frame chunk or JSON control message
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                # Process audio chunk
                result = pipeline.process_incoming_event(session, "audio_chunk", data["bytes"])
                await websocket.send_json(result)
            elif "text" in data and data["text"]:
                import json
                try:
                    payload = json.loads(data["text"])
                    event_type = payload.get("type", "commit_turn")
                    result = pipeline.process_incoming_event(session, event_type, b"")
                    await websocket.send_json(result)
                except Exception:
                    await websocket.send_json({"error": "Invalid JSON format"})
    except WebSocketDisconnect:
        if session_id in active_sessions:
            del active_sessions[session_id]
