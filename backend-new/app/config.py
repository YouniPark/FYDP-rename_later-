from pathlib import Path

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

    project_root: Path = Path(__file__).resolve().parents[2]

    # webserver directory 
    webserver_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "WebServer",
        alias="WEBSERVER_DIR",
    )

    # PeopleDatabase directory (where all the data - auditory cues, images, headshots, and information)
    people_database_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "WebServer"
        / "database"
        / "PeopleDatabase",
        alias="PEOPLE_DATABASE_DIR",
    )

    # people.json file where all the information is stored 
    people_json_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "WebServer"
        / "database"
        / "PeopleDatabase"
        / "people.json",
        alias="PEOPLE_JSON_PATH",
    )

    # images folder directory where all the cue images are stored 
    images_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "WebServer"
        / "database"
        / "PeopleDatabase"
        / "images",
        alias="IMAGES_DIR",
    )

    # auditory cues directory where all the auditory cues are stored 
    auditory_cue_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "WebServer"
        / "database"
        / "PeopleDatabase"
        / "auditory cues",
        alias="AUDITORY_CUE_DIR",
    )

    # headshots folder directory where all the headshot videos are stored (for facial recognition)
    headshots_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "WebServer"
        / "database"
        / "PeopleDatabase"
        / "headshots",
        alias="HEADSHOTS_DIR",
    )

    # setting directory
    setting_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "WebServer"
        / "setting",
        alias="SETTING_DIR",
    )


settings = Settings()