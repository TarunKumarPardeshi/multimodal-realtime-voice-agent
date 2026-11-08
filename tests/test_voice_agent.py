"""
Unit and Integration Tests for Real-Time Multimodal Voice Agent
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.voice_agent.audio_vad import AudioVADProcessor
from src.voice_agent.vision_processor import VisionProcessor
from src.voice_agent.session_manager import StreamingSession
from src.voice_agent.agent_pipeline import MultimodalAgentPipeline
from src.voice_agent.api import app

client = TestClient(app)

def test_audio_vad_energy():
    vad = AudioVADProcessor(energy_threshold=0.015)
    # Silence
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    is_speech, barge_in = vad.process_chunk(silence)
    assert is_speech is False
    assert barge_in is False

    # Loud speech
    loud = (np.ones(1600, dtype=np.int16) * 15000).tobytes()
    is_speech_loud, barge_in_loud = vad.process_chunk(loud)
    assert is_speech_loud is True
    assert barge_in_loud is True

def test_vision_processor_keyframe():
    vp = VisionProcessor()
    frame1 = b"\xFF\xD8\xFF\xE0" + b"\x01" * 100
    info1 = vp.process_frame(frame1)
    assert info1["is_keyframe"] is True

    # Same frame again
    info2 = vp.process_frame(frame1)
    assert info2["is_keyframe"] is False

def test_multimodal_pipeline_and_barge_in():
    pipeline = MultimodalAgentPipeline()
    session = StreamingSession("test-session")

    # Ingest video frame
    frame = b"\xFF\xD8\xFF\xE0" + b"\x02" * 500
    res_v = pipeline.process_incoming_event(session, "video_frame", frame)
    assert res_v["event"] == "frame_ingested"

    # Commit turn
    res_turn = pipeline.process_incoming_event(session, "commit_turn", b"")
    assert res_turn["event"] == "agent_response"
    assert session.agent_is_speaking is True

    # Barge-in interrupt
    loud_audio = (np.ones(1600, dtype=np.int16) * 18000).tobytes()
    res_barge = pipeline.process_incoming_event(session, "audio_chunk", loud_audio)
    assert res_barge["event"] == "barge_in_interruption"
    assert session.agent_is_speaking is False

def test_api_health_and_session_start():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    start_resp = client.post("/api/v1/session/start")
    assert start_resp.status_code == 200
    data = start_resp.json()
    assert "session_id" in data
    assert data["sample_rate"] == 16000
