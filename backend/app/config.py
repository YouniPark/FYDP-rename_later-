from __future__ import annotations

import os

APP_NAME = os.getenv("APP_NAME", "python-backend-server")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DB_PATH = os.getenv("DB_PATH", "backend.db")

EEG_RETRY_SECONDS = float(os.getenv("EEG_RETRY_SECONDS", "3.0"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
ALLOWED_IMAGE_FORMATS = {"png", "jpg", "jpeg"}
