from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------
    # General / storage
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # EEG Config
    # ------------------------------------------------------------------

    eeg_lsl_retry_seconds: int = Field(default=3, alias="EEG_LSL_RETRY_SECONDS")

    # Bandpass filter bounds (Hz)
    eeg_l_freq: float = Field(default=1.0, alias="EEG_L_FREQ")
    eeg_h_freq: float = Field(default=40.0, alias="EEG_H_FREQ")

    # Notch filter frequencies as a JSON array, e.g. "[50.0, 100.0]" for 50 Hz regions
    eeg_notch_freqs: list[float] = Field(
        default_factory=lambda: [60.0, 120.0],
        alias="EEG_NOTCH_FREQS",
    )

    # Path to a saved MNE ICA .fif file; leave unset to skip ICA removal
    eeg_ica_path: Optional[str] = Field(default='./data/models/ica.fif', alias="EEG_ICA_PATH")

    # Whether to apply REST re-referencing
    eeg_apply_rest: bool = Field(default=True, alias="EEG_APPLY_REST")

    # Path to a pre-computed forward solution for REST; required when eeg_apply_rest=True
    eeg_forward_path: Optional[str] = Field(default='./data/models/rest_forward.fif', alias="EEG_FORWARD_PATH")

    # Baseline correction window (seconds relative to event).
    # eeg_baseline_tmin=None means "from epoch start".
    eeg_baseline_tmin: Optional[float] = Field(default=-1.0, alias="EEG_BASELINE_TMIN")
    eeg_baseline_tmax: float = Field(default=-0.7, alias="EEG_BASELINE_TMAX")

    # Peak-to-peak artifact rejection threshold in microvolts
    eeg_amp_thresh_uv: float = Field(default=200.0, alias="EEG_AMP_THRESH_UV")

    # Path to the trained SVM model and fitted scaler (.joblib files)
    eeg_model_path: str = Field(default="./data/models/erp_svm_model.joblib", alias="EEG_MODEL_PATH")
    eeg_scaler_path: str = Field(default="./data/models/erp_feature_scaler.joblib", alias="EEG_SCALER_PATH")


settings = Settings()