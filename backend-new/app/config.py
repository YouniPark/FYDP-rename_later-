from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    eeg_lsl_retry_seconds: int = Field(default=3, alias="EEG_LSL_RETRY_SECONDS")
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    face_images_dir: str = Field(default="./data/faces", alias="FACE_IMAGES_DIR")
    cue_images_dir: str = Field(default="./data/cues", alias="CUE_IMAGES_DIR")
    video_mode: str = Field(default="ws", alias="VIDEO_MODE")
    video_pull_url: str | None = Field(default=None, alias="VIDEO_PULL_URL")
    max_frame_queue: int = Field(default=32, alias="MAX_FRAME_QUEUE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_upload_bytes: int = Field(default=5_000_000, alias="MAX_UPLOAD_BYTES")


settings = Settings()
