"""Download third-party pretrained YOLOv8 road-damage weights from Hugging Face.

Weights have their own license. Inspect model card at:
https://huggingface.co/dronefreak/rdd2022-yolov8s
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "dronefreak/rdd2022-yolov8s"
FILENAME = "best.pt"


def download_weights(target_dir: str = "models") -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install requirements: python -m pip install -r requirements.txt") from exc
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(dest),
    )
    return Path(result)


def main():
    parser = argparse.ArgumentParser(description="Get the PavePulse road damage YOLOv8s model")
    parser.add_argument("--target-dir", default="models")
    args = parser.parse_args()
    print(f"Downloading model from https://huggingface.co/{REPO_ID} ...")
    try:
        path = download_weights(args.target_dir)
    except Exception as exc:
        print(f"ERROR: Could not download the model: {exc}")
        print("Check your internet connection or manually download best.pt from the model's Files tab.")
        raise SystemExit(1) from exc
    print(f"Model ready: {path.resolve()}")


if __name__ == "__main__":
    main()
