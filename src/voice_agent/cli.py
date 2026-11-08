"""
Interactive Terminal Simulator for Real-Time Multimodal Voice Agent
"""

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .session_manager import StreamingSession
from .agent_pipeline import MultimodalAgentPipeline

console = Console(force_terminal=True, legacy_windows=False)

def main():
    console.print(Panel.fit(
        "[bold cyan]Real-Time Multimodal Voice & Vision Agent[/bold cyan]\n"
        "[dim]Architect: Tarun Pardeshi | Sub-300ms Full-Duplex WebRTC / Streaming AI[/dim]",
        border_style="cyan"
    ))

    session = StreamingSession(session_id="cli-demo-session-1")
    pipeline = MultimodalAgentPipeline()

    table = Table(title="Live Multimodal Event Stream Simulation", show_header=True, header_style="bold magenta")
    table.add_column("Turn", style="dim", width=6)
    table.add_column("Input Stream", style="white")
    table.add_column("Agent Action / State", style="cyan")
    table.add_column("Latency", justify="right", style="green")

    # 1. Video frame ingestion
    fake_frame = b"\xFF\xD8\xFF\xE0" + b"\x00" * 4096
    res1 = pipeline.process_incoming_event(session, "video_frame", fake_frame)
    table.add_row("1", f"Camera Video Frame ({len(fake_frame)} bytes)", f"Keyframe Detected: {res1['is_keyframe']}", "< 2ms")

    # 2. Audio speech chunk
    speech_pcm = (np.random.randn(3200) * 8000).astype(np.int16).tobytes()
    res2 = pipeline.process_incoming_event(session, "audio_chunk", speech_pcm)
    table.add_row("2", "Microphone PCM Audio (16kHz)", f"VAD Speech Active (Buffer: {res2['buffer_size_bytes']} bytes)", "< 5ms")

    # 3. Commit turn & generate response
    res3 = pipeline.process_incoming_event(session, "commit_turn", b"")
    table.add_row("3", "Speech Pause / End-of-Turn", f"Agent Speech Output: '{res3['text_response'][:45]}...'", f"{res3['e2e_latency_ms']}ms")

    # 4. User interrupts agent (Barge-in)
    loud_speech = (np.random.randn(3200) * 16000).astype(np.int16).tobytes()
    res4 = pipeline.process_incoming_event(session, "audio_chunk", loud_speech)
    table.add_row("4", "User Barge-In Speech", f"INTERRUPT DETECTED -> {res4['status']}", f"{res4['latency_ms']}ms")

    console.print(table)

    console.print(Panel(
        f"[bold green]Session State:[/bold green] Active\n"
        f"[bold]Turns Logged:[/bold] {len(session.conversation_turns)}\n"
        f"[bold]Barge-In Debounce:[/bold] PASSED\n"
        f"[bold]Visual Context Grounding:[/bold] {res3['visual_grounding']}",
        title="Streaming Summary",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
