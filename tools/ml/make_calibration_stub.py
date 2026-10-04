"""
Builds onnx2tf's representative-dataset calibration file from real photos, not random noise.

onnx2tf's convert() uses calibration_image_sample_data_20x128x128x3_float32.npy as its default
representative dataset for int8 post-training quantization -- it runs these images through the
model to measure real activation ranges, which is what the int8 scale/zero-point for every
tensor actually gets calibrated against. onnx2tf normally downloads its own copy of this file
from an S3 bucket (onnx2tf-en) that is now permanently gone (404 NoSuchBucket), which is why a
local stand-in is needed at all -- but the first version of this script filled that stand-in
with uniform random pixel noise, on the mistaken assumption that only the shape mattered. It
doesn't: calibrating against noise (which has none of a real photo's spatial structure or
intensity distribution) produces degenerate int8 ranges that collapse real-world model output to
a near-constant signal. Confirmed on-device: every class score and box coordinate decoded to the
exact same value on every frame, regardless of what the camera saw.

This version builds the same required (20, 128, 128, 3) float32 shape from 20 real photos
(Ultralytics' own coco128 sample set -- see tools/ml/README.md for how to fetch it), resized and
center-cropped to 128x128 like any other ImageNet-style preprocessing, so onnx2tf calibrates
against activations a real camera frame would actually produce.

Pixel values are normalized to [0, 1] (divided by 255) before saving. This matters as much as
using real photos does: inspecting the exported ONNX graph directly (see tools/ml/README.md)
shows its first op is a Conv applied straight to the "images" input -- there's no Div-by-255
node baked into the graph, meaning Ultralytics' export expects already-normalized [0,1] float32
input (the /255 happens in Python-side preprocessing, not in the graph). The first version of
this script (and its random-noise predecessor) fed onnx2tf raw 0-255 pixel values, 255x outside
the range the model's input tensor will ever actually see at inference time. onnx2tf calibrated
the int8 input scale/zero-point against that wrong range, so every real (correctly 0-1
normalized) camera frame quantized into a single saturated int8 bucket -- a constant input
regardless of what the camera saw, which is exactly the frozen/zero-confidence output observed
on-device.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FILENAME = "calibration_image_sample_data_20x128x128x3_float32.npy"
TARGET_SIZE = 128
NUM_IMAGES = 20


def load_and_center_crop(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    images_dir = Path("coco128/images/train2017")
    if not images_dir.is_dir():
        sys.exit(
            f"Expected real sample images at {images_dir}/ (see tools/ml/README.md's coco128 "
            "download step) -- not found. Refusing to fall back to synthetic data; that's the "
            "exact bug this script fixes."
        )

    image_paths = sorted(images_dir.glob("*.jpg"))[:NUM_IMAGES]
    if len(image_paths) < NUM_IMAGES:
        sys.exit(f"Only found {len(image_paths)} images in {images_dir}, need {NUM_IMAGES}.")

    batch = np.stack([load_and_center_crop(p, TARGET_SIZE) for p in image_paths])
    assert batch.shape == (NUM_IMAGES, TARGET_SIZE, TARGET_SIZE, 3), batch.shape

    np.save(FILENAME, batch, allow_pickle=False)
    print(f"Wrote {FILENAME} from {len(image_paths)} real images: {[p.name for p in image_paths]}")


if __name__ == "__main__":
    main()
