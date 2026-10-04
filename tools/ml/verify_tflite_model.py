"""
Inspect a .tflite model's input/output tensor shape, dtype, and
quantization (scale/zero-point) before trusting it enough to bundle
into the Android app.

Usage:
    python verify_tflite_model.py yolov8n_saved_model/yolov8n_full_integer_quant.tflite

Only accept the file if BOTH input and output dtype print as int8 (or
uint8) -- that's what "full integer quant" is supposed to mean, but it's
worth confirming rather than trusting the filename alone.

The printed output shape and quantization (scale, zero_point) must be
copied into the Kotlin decoder in ObjectDetectorHelper.kt: onnx2tf
(which Ultralytics' TFLite export uses internally) commonly transposes
the output to channel-last, so don't assume [1, 84, 8400] -- it may
print as [1, 8400, 84] instead.
"""
import sys

import tensorflow as tf


def inspect(path: str) -> None:
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"Model: {path}")
    print(f"  input : shape={in_d['shape']} dtype={in_d['dtype']} quant={in_d['quantization']}")
    print(f"  output: shape={out_d['shape']} dtype={out_d['dtype']} quant={out_d['quantization']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_tflite_model.py <path-to-tflite-model>")
        sys.exit(1)
    inspect(sys.argv[1])
