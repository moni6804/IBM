import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("MODEL_FILE", str(ROOT / "data" / "test_model.joblib"))
for key in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID"):
    os.environ.pop(key, None)  # tests always run offline
