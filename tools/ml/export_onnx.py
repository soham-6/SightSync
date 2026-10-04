"""
Step 1 of 2: export pretrained YOLOv8n (.pt) -> ONNX, plus labels.txt.

Runs anywhere (Windows/Linux/macOS) -- this part needs no special environment.
Step 2 (ONNX -> int8 TFLite) must run on Linux; see README.md for why and
the exact Docker command.

    conda activate sightsync-ml   # or any env with `pip install ultralytics`
    cd tools/ml
    python export_onnx.py
"""
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent


def main() -> None:
    model = YOLO(str(HERE / "yolov8n.pt"))  # auto-downloads pretrained weights if absent

    onnx_path = model.export(format="onnx", imgsz=640)
    print(f"Exported ONNX: {onnx_path}")

    labels_out = HERE / "labels.txt"
    with open(labels_out, "w") as f:
        for idx in sorted(model.names.keys()):
            f.write(f"{model.names[idx]}\n")
    print(f"Wrote {labels_out}")


if __name__ == "__main__":
    main()
