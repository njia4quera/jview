# producer.py
import os, requests, numpy as np
from datetime import datetime
from app import frames_to_b64_npz  # or copy these 12 lines into your project

APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8050")
TOKEN = os.getenv("INGEST_TOKEN", "")  # must match server if set

def send_shot_http(frames: dict[str, np.ndarray], meta: dict):
    payload = {
        "shot_name": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta,
        "frames_npz_b64": frames_to_b64_npz(frames),
    }
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["X-Ingest-Token"] = TOKEN
    r = requests.post(f"{APP_URL}/api/add_shot", json=payload, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    rng = np.random.default_rng()
    frames = {
        "raw": rng.normal(1000, 40, size=(128,128)).astype(np.float32),
        "proc": rng.normal(1000, 40, size=(128,128)).astype(np.float32)
        }
    meta = {"detuning_MHz": np.random.rand(), "sequence": "exp2"}
    print(send_shot_http(frames, meta))
