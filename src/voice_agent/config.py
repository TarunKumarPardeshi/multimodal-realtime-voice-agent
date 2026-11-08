from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    environment: str = "development"

    # Audio & Voice Activity Detection
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    vad_energy_threshold: float = 0.015
    barge_in_debounce_ms: int = 120

    # Vision
    max_video_frame_fps: int = 2
    image_resize_max_width: int = 1024

    # Bedrock / LLM
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
