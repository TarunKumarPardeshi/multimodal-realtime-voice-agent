"""
Voice Activity Detection (VAD) and Barge-In Audio Processor
Detects speech onset, pauses, and handles natural conversational interruption.
"""

import numpy as np
from typing import Tuple

class AudioVADProcessor:
    def __init__(self, sample_rate: int = 16000, energy_threshold: float = 0.015):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.is_speaking = False
        self.silence_frame_count = 0

    def compute_frame_energy(self, pcm_bytes: bytes) -> float:
        """
        Computes Root Mean Square (RMS) energy across 16-bit PCM audio bytes.
        """
        if not pcm_bytes:
            return 0.0
        # Convert raw bytes to int16 numpy array
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return 0.0
        # Normalize to [-1.0, 1.0]
        samples = samples / 32768.0
        rms = np.sqrt(np.mean(samples**2))
        return float(rms)

    def process_chunk(self, pcm_bytes: bytes) -> Tuple[bool, bool]:
        """
        Processes an incoming streaming audio chunk.
        Returns: (speech_detected, barge_in_triggered)
        """
        energy = self.compute_frame_energy(pcm_bytes)
        barge_in = False

        if energy > self.energy_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                barge_in = True  # User started talking while agent might be speaking
            self.silence_frame_count = 0
            return True, barge_in
        else:
            self.silence_frame_count += 1
            if self.silence_frame_count > 10:  # ~200ms of silence
                self.is_speaking = False
            return False, False
