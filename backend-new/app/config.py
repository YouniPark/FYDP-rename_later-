from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import socket

def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

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

    eeg_lsl_stream_name: str = Field(default="Explore_84A1_ExG", alias="EEG_LSL_STREAM_NAME")
    eeg_lsl_retry_seconds: int = Field(default=6.0, alias="EEG_LSL_RETRY_SECONDS")

    # EEG channel names in device output order (positional mapping from stream channels to EEG names).
    # Set EEG_CHANNEL_NAMES as a JSON array in your .env to override, e.g.:
    #   EEG_CHANNEL_NAMES=["Cz","Pz","P8","P7","O2","O1"]
    eeg_channel_names: list[str] = Field(
        default_factory=lambda: ["Cz", "Pz", "P8", "P7", "O2", "O1"],
        alias="EEG_CHANNEL_NAMES",
    )

    # Bandpass filter bounds (Hz)
    eeg_l_freq: float = Field(default=1.0, alias="EEG_L_FREQ")
    eeg_h_freq: float = Field(default=40.0, alias="EEG_H_FREQ")

    # Notch filter frequencies as a JSON array, e.g. "[50.0, 100.0]" for 50 Hz regions
    eeg_notch_freqs: list[float] = Field(
        default_factory=lambda: [60.0, 120.0],
        alias="EEG_NOTCH_FREQS",
    )

    # Path to a saved MNE ICA .fif file; leave unset to skip ICA removal
    eeg_ica_path: Optional[str] = Field(default='./data/models/eeg-ica.fif', alias="EEG_ICA_PATH")

    # Whether to apply REST re-referencing
    eeg_apply_rest: bool = Field(default=True, alias="EEG_APPLY_REST")

    # Path to a pre-computed forward solution for REST; required when eeg_apply_rest=True
    eeg_forward_path: Optional[str] = Field(default='./data/models/rest-fwd.fif', alias="EEG_FORWARD_PATH")

    # Baseline correction window (seconds relative to event).
    # eeg_baseline_tmin=None means "from epoch start".
    eeg_baseline_tmin: Optional[float] = Field(default=-1.0, alias="EEG_BASELINE_TMIN")
    eeg_baseline_tmax: float = Field(default=-0.7, alias="EEG_BASELINE_TMAX")

    # Epoch window (seconds relative to event timestamp).
    # tmax controls how long after the event we must wait before reading the EEG buffer.
    eeg_epoch_tmin: float = Field(default=-1.0, alias="EEG_EPOCH_TMIN")
    eeg_epoch_tmax: float = Field(default=0.65, alias="EEG_EPOCH_TMAX")

    # How often to poll the EEG buffer while waiting for post-event data to arrive (seconds).
    eeg_buffer_poll_interval: float = Field(default=0.1, alias="EEG_BUFFER_POLL_INTERVAL")
    # Maximum total time to wait for the buffer to cover the epoch window before giving up (seconds).
    eeg_buffer_poll_timeout: float = Field(default=10.0, alias="EEG_BUFFER_POLL_TIMEOUT")

    # Peak-to-peak artifact rejection threshold in microvolts
    eeg_amp_thresh_uv: float = Field(default=200.0, alias="EEG_AMP_THRESH_UV")

    # When True, proceed to feature extraction and ML classification even if the epoch is
    # rejected by the artifact check. Bad channels are imputed from the group average (N250/P300).
    # Set IGNORE_TRIAL_REJECTION=true in .env to enable.
    ignore_trial_rejection: bool = Field(default=False, alias="IGNORE_TRIAL_REJECTION")

    # Path to the trained SVM model and fitted scaler (.joblib files)
    eeg_model_path: str = Field(default="./data/models/erp_svm_model.joblib", alias="EEG_MODEL_PATH")
    eeg_scaler_path: str = Field(default="./data/models/erp_feature_scaler.joblib", alias="EEG_SCALER_PATH")

    # ------------------------------------------------------------------
    # ERP Feature Logging (data collection)
    # ------------------------------------------------------------------

    # Set to True to append ERP feature vectors to CSV files after each pipeline run.
    # Useful for accumulating labelled data for future model training.
    eeg_save_erp_features: bool = Field(default=True, alias="EEG_SAVE_ERP_FEATURES")

    # CSV path for raw (pre-scaled) feature vectors
    eeg_features_raw_csv_path: str = Field(
        default="./data/generated/erp_features_raw.csv", alias="EEG_FEATURES_RAW_CSV_PATH"
    )

    # CSV path for scaled (post-scaler) feature vectors
    eeg_features_scaled_csv_path: str = Field(
        default="./data/generated/erp_features_scaled.csv", alias="EEG_FEATURES_SCALED_CSV_PATH"
    )

    # ------------------------------------------------------------------
    # LSL Fixation Inlet
    # ------------------------------------------------------------------

    # Name of the LSL stream created by FaceProxyGazeInteractor in Unity
    lsl_fixation_stream_name: str = Field(default="FixationEvents", alias="LSL_FIXATION_STREAM_NAME")
    # How long (seconds) to wait when resolving the stream on the network
    lsl_fixation_resolve_timeout: float = Field(default=5.0, alias="LSL_FIXATION_RESOLVE_TIMEOUT")
    # Polling interval (seconds) between non-blocking pull_sample calls
    lsl_fixation_poll_interval: float = Field(default=0.1, alias="LSL_FIXATION_POLL_INTERVAL")
    # How long to wait before retrying after a connection failure or stream loss
    lsl_fixation_retry_seconds: float = Field(default=6.0, alias="LSL_FIXATION_RETRY_SECONDS")

    # Minimum seconds between accepted fixation events.  Events arriving sooner
    # than this after the last accepted event are silently dropped at the inlet.
    event_min_interval_s: float = Field(default=3.0, alias="EVENT_MIN_INTERVAL_S")

    # If True, events are processed as "unfamiliar" when the EEG stream is not connected.
    # If False, events are silently dropped until the EEG stream becomes available.
    unfamiliar_if_no_eeg: bool = Field(default=True, alias="UNFAMILIAR_IF_NO_EEG")

    # ------------------------------------------------------------------
    # Cue Delivery
    # ------------------------------------------------------------------

    # When True, the backend sends only the people_id in the cue payload.
    # The AR app loads name/image/audio from its own local cue JSON files.
    # When False, the backend reads settings.json and sends full cue data
    # (including binary image/audio bytes) in the payload.
    use_local_cues: bool = Field(default=True, alias="USE_LOCAL_CUES")

    # ------------------------------------------------------------------
    # Face Recognition Config
    # ------------------------------------------------------------------
    memory_window_seconds: float = 5.0
    memory_min_votes: int = 2
    memory_majority_ratio: float = 0.6

    # ------------------------------------------------------------------
    # Cues to URL
    # ------------------------------------------------------------------
    public_url: str = Field(
        default_factory=lambda: f"http://{get_lan_ip()}:8000/media",
        alias="PUBLIC_URL",
    )

    use_url: bool = Field(default=True, alias="USE_URL")

settings = Settings()