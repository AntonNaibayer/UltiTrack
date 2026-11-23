from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw"
PROCESSED_VIDEOS_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

MODEL_NAME = "ulti_indoor_with_disk_v3.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME

# Пороги уверенности (confidence)
CONFIDENCE_THRESHOLD = 0.15       # Для игроков
DISC_CONFIDENCE_THRESHOLD = 0.22  # Для диска на полном кадре
DISC_CROP_CONFIDENCE = 0.14       # Для поиска диска в кропе вокруг частиц

IMAGE_SIZE = 1280

# 0 — disc, 1 — player
TARGET_CLASSES = [0, 1]

# Размеры поля в метрах (можно менять под конкретный зал)
FIELD_WIDTH = 20.0
FIELD_LENGTH = 41.0
ENDZONE_LENGTH = 7.0