from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
GEOJSON_DIR = DATA_DIR / "geojson"
MODEL_DIR = PROJECT_ROOT / "Model"

GEODATA_DIR = PROJECT_ROOT / "Geodata Jawa Tengah"
JAWA_TENGAH_SHAPEFILE = GEODATA_DIR / "JawaTengah.shp"
JAWA_TENGAH_GEOJSON_CACHE = GEOJSON_DIR / "jawa_tengah.geojson"
INDONESIA_GEOJSON = GEOJSON_DIR / "indonesia_regencies.geojson"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000

CORS_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]
