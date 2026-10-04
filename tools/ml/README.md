# SightSync — YOLOv8n → TFLite conversion (Phase 2/3)

One-time, offline process that produces the `.tflite` model bundled into the Android app at
`app/src/main/assets/yolov8n_float32.tflite`. None of this runs on the phone.

**Currently bundled: float32, not int8.** int8 export is still produced by the pipeline below
(and verified to load/run on-device) but is not usable yet -- see "Known issue: int8 output
quantization" at the bottom before reviving it.

It's a **two-step** pipeline across two different environments:

1. **ONNX export** (`export_onnx.py`) — runs anywhere, including Windows.
2. **ONNX → int8 TFLite conversion** (`onnx2tf`) — must run on **Linux**. On Windows this
   consistently crashed with no usable error across every version combination we tried; the
   same crash reproduced identically inside a Linux Docker container until an unrelated dead
   upstream dependency (below) was fixed. We used Docker to get a Linux environment without
   needing WSL installed.

## Step 1 — Export to ONNX (Windows or Linux)

```bash
conda create -n sightsync-ml python=3.11 -y
conda activate sightsync-ml
pip install -r requirements.txt
cd tools/ml
python export_onnx.py
```

Produces `yolov8n.onnx` and `labels.txt` (COCO-80 class names, taken directly from the
loaded model's own class list so label order is guaranteed to match the model's output
class-index order — never hand-copy a COCO list from the web).

Python 3.11 is used deliberately — not the base Anaconda Python 3.13 — since the newer
toolchain in step 2 is better tested against it.

## Step 2 — Convert to TFLite (must be Linux — use Docker on Windows)

### Why Linux only

Two separate things went wrong here, in order:

1. **Ultralytics itself can't export TFLite on Windows anymore.** Ultralytics 8.4.x replaced
   its TFLite export with a new "LiteRT" path that asserts `Linux x86 or macOS` and refuses
   to run on Windows at all. (It also pulls in a huge, fragile dependency chain — JAX,
   HuggingFace transformers, etc. — and crashed with a `torch`/`torchvision` version conflict
   when we tried it in a container. Avoid this path; use `onnx2tf` directly instead, as below.)
2. **`onnx2tf` (the community ONNX→TFLite converter) crashed identically on both Windows and
   Linux** — same silent death immediately after `Model conversion started`, across 4+
   onnx2tf versions, 3 TensorFlow versions, 2 ONNX opsets, with/without multithreading. The
   real cause (visible once captured through a plain `.NET Process` invocation that didn't
   swallow stderr the way piping through nested shells did) was:

   ```
   ValueError: Cannot load file containing pickled data when allow_pickle=False
   ```

   raised from an **unguarded** second call to onnx2tf's `download_test_image_data()`. That
   function tries to download a small calibration image sample from onnx2tf's own S3 bucket
   (`onnx2tf-en`) — which returns `404 NoSuchBucket`. The bucket is just gone. This breaks
   *every* onnx2tf conversion run, on any OS, regardless of version, whenever no local copy
   of that file already exists. It's unrelated to the "Linux only" LiteRT restriction above —
   purely a coincidence that both problems hit at once.

   **Fix:** run `make_calibration_stub.py` first (see below) to create a stand-in file with
   the exact filename/shape onnx2tf expects in the current working directory, so it's found
   locally and the dead download is never attempted.

### Commands (via Docker, from the repo's `tools/ml/` directory)

`make_calibration_stub.py` needs real sample images to build onnx2tf's representative dataset
(see "Known issue" below for why this matters) -- fetch Ultralytics' coco128 sample set first:

```powershell
curl -L -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
Expand-Archive coco128.zip -DestinationPath .   # or `unzip coco128.zip`

python make_calibration_stub.py

docker run --rm -v C:\SightSync\tools\ml:/workspace -w /workspace `
  -e TF_USE_LEGACY_KERAS=1 python:3.11-slim bash -c `
  "pip install --quiet tensorflow tf-keras onnx onnx2tf onnxsim sng4onnx onnx_graphsurgeon ai-edge-litert psutil && python -m onnx2tf -i yolov8n.onnx -o yolov8n_saved_model -oiqt -qt per-tensor"
```

This produces `yolov8n_saved_model/` containing several `.tflite` variants:

- `yolov8n_float32.tflite` — **this is the one currently bundled into the app**
- `yolov8n_float16.tflite`
- `yolov8n_dynamic_range_quant.tflite`
- `yolov8n_integer_quant.tflite` — int8 weights, float32 I/O
- `yolov8n_full_integer_quant.tflite` — int8 weights **and** int8 input/output (not usable yet — see "Known issue" below)

If this is ever run on a machine *without* Docker: any Linux box or VM works the same way —
just run the same `pip install ...` + `python -m onnx2tf ...` commands directly, with
`make_calibration_stub.py` run first in that same working directory.

## Step 3 — Verify before trusting the output

```bash
python verify_tflite_model.py yolov8n_saved_model/yolov8n_float32.tflite
```

Confirm input/output `dtype` print as `float32` and shape is `[1, 84, 8400]` (or `[1, 8400, 84]`
— onnx2tf's channel ordering isn't guaranteed; `ObjectDetectorHelper.kt` reads this at runtime
rather than hardcoding it).

## Step 4 — Bundle into the app

```bash
cp yolov8n_saved_model/yolov8n_float32.tflite ../../app/src/main/assets/yolov8n_float32.tflite
cp labels.txt ../../app/src/main/assets/labels.txt
```

## Known issue: int8 output quantization (why float32 is bundled instead)

`yolov8n_full_integer_quant.tflite` loads and runs fine on-device but produces unusable output:
every class score decodes to <= 0 on every single frame regardless of what the camera sees.
Root-caused on a real phone (Nothing Phone 4a) via two layers:

1. **Fixed:** `make_calibration_stub.py`'s representative dataset (used by onnx2tf to compute
   int8 scale/zero-point) was built from either random noise or raw 0-255 pixel values.
   Inspecting the exported ONNX graph directly shows its first op is a `Conv` applied straight
   to the `images` input — no `Div`-by-255 node is baked in, meaning the model expects
   already-normalized `[0,1]` float input (Ultralytics does the `/255` in Python-side
   preprocessing, not in the graph). Calibrating against data 255x outside that range produced
   a badly-scaled quantization. Fixed by normalizing calibration images to `[0,1]` before
   saving (see the script) — this noticeably changed the calibrated scale (confirmed via
   `verify_tflite_model.py`) but did not fix the actual symptom.
2. **Not yet fixed:** the real cause. `-qt per-tensor` applies **one shared** scale/zero-point
   across the *entire* `[1, 84, 8400]` output tensor, which packs together 4 box-coordinate
   channels (range 0-640 pixels) and 80 class-score channels (much smaller logit range) with no
   per-channel distinction. A single scale wide enough to represent box coordinates is far too
   coarse to represent the class-score range with any useful precision, so real class scores
   collapse into 1-2 indistinguishable int8 buckets near the low end — confirmed by swapping in
   the float32 model with no other code changes and immediately getting correct, confident,
   frame-to-frame-consistent detections.

Properly fixing this means re-exporting with box coordinates and class scores as two *separate*
output tensors, each getting its own independent quantization range, before running them through
onnx2tf -- not just a calibration-data fix. Deferred as a later performance pass (the payoff is
NNAPI-accelerated real-time inference; float32 currently runs ~400ms/frame on CPU alone, no
NNAPI benefit observed for float32 on this hardware).
