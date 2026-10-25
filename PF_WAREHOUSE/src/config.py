from pathlib import Path

# Rutas principales del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "train.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Hiperparámetros y configuración de entrenamiento
SEED = 42
TEST_SIZE = 0.2
QUALITY_GATE_RMSE = 15000.0  # Puerta de calidad: RMSE máximo permitido