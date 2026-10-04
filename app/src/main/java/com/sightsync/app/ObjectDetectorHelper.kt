package com.sightsync.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Runs a YOLOv8n TFLite model (converted offline via tools/ml/) on a single frame. Reads
 * quantization params and output tensor shape from the model itself at runtime rather than
 * hardcoding them, since onnx2tf's channel ordering and quant scheme can vary between conversion
 * runs; for a float32 model (the current default) those params are simply unused.
 *
 * Bundled as float32, not int8, despite int8 being the original plan: onnx2tf's `-qt per-tensor`
 * forces a single shared scale/zero-point across the whole [box_coords, class_scores] output
 * tensor. Box coordinates span the full 0-640 pixel range while class-score logits span a much
 * smaller range, so a shared per-tensor scale calibrated for the former crushes the latter into
 * a handful of indistinguishable quantization buckets -- confirmed on-device: every class score
 * decoded to <= 0 on every frame regardless of what the camera saw, even after fixing an
 * unrelated calibration-data bug (tools/ml/make_calibration_stub.py was feeding onnx2tf raw
 * 0-255 pixel values against a model whose ONNX graph expects pre-normalized [0,1] input).
 * Properly fixing int8 would mean re-exporting with box coords and class scores as separate
 * output tensors so each gets its own quantization range -- deferred as a later performance
 * pass; see tools/ml/README.md.
 */
class ObjectDetectorHelper(
    context: Context,
    modelAssetPath: String = "yolov8n_float32.tflite",
    labelsAssetPath: String = "labels.txt",
    private val inputSize: Int = 640,
    private val confidenceThreshold: Float = 0.4f,
    private val iouThreshold: Float = 0.45f
) {
    data class Detection(
        val classId: Int,
        val label: String,
        val confidence: Float,
        val box: RectF
    )

    private val labels: List<String> =
        context.assets.open(labelsAssetPath).bufferedReader().readLines().filter { it.isNotBlank() }

    private var activeDelegate: Closeable? = null
    private val interpreter: Interpreter = buildInterpreter(context, modelAssetPath)

    private val inputDataType: DataType = interpreter.getInputTensor(0).dataType()
    private val inputQuant = interpreter.getInputTensor(0).quantizationParams()
    private val outputDataType: DataType = interpreter.getOutputTensor(0).dataType()
    private val outputQuant = interpreter.getOutputTensor(0).quantizationParams()
    private val outputShape: IntArray = interpreter.getOutputTensor(0).shape() // [1,84,N] or [1,N,84]

    private val numAttrs = labels.size + 4
    private val channelsFirst = outputShape.size == 3 && outputShape[1] == numAttrs
    private val numAnchors = if (channelsFirst) outputShape[2] else outputShape[1]

    init {
        Log.i(
            TAG,
            "Model loaded. input dtype=$inputDataType quant=(scale=${inputQuant.scale}, zp=${inputQuant.zeroPoint}) " +
                "output shape=${outputShape.joinToString()} dtype=$outputDataType " +
                "quant=(scale=${outputQuant.scale}, zp=${outputQuant.zeroPoint}) channelsFirst=$channelsFirst"
        )
    }

    private fun buildInterpreter(context: Context, modelAssetPath: String): Interpreter {
        val model = loadModelFile(context, modelAssetPath)

        // GPU delegate first: NNAPI's benefit is specific to int8 models (see the class doc for
        // why this model is float32), but a float32 CNN like this typically runs substantially
        // faster on the mobile GPU than on CPU. CompatibilityList steers away from devices/driver
        // combos TFLite knows mishandle it, so this only engages where it's expected to work.
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                val delegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                val interp = Interpreter(model, Interpreter.Options().addDelegate(delegate))
                activeDelegate = delegate
                Log.i(TAG, "Using GPU delegate")
                return interp
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate failed, falling back to NNAPI/CPU", e)
            }
        }

        val options = Interpreter.Options().apply { setNumThreads(4) }
        try {
            val delegate = NnApiDelegate()
            options.addDelegate(delegate)
            activeDelegate = delegate
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI delegate unavailable, using CPU", e)
        }
        return try {
            val interp = Interpreter(model, options)
            Log.i(TAG, if (activeDelegate != null) "Using NNAPI delegate" else "Using CPU (no delegate)")
            interp
        } catch (e: Exception) {
            Log.w(TAG, "Interpreter init with NNAPI failed, retrying CPU-only", e)
            activeDelegate?.close()
            activeDelegate = null
            Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
        }
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).use { input ->
            val channel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    fun detect(bitmap: Bitmap, rotationDegrees: Int): List<Detection> {
        val rotated = if (rotationDegrees != 0) {
            val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
        } else {
            bitmap
        }

        val scale = min(inputSize.toFloat() / rotated.width, inputSize.toFloat() / rotated.height)
        val scaledW = (rotated.width * scale).roundToInt()
        val scaledH = (rotated.height * scale).roundToInt()
        val padX = (inputSize - scaledW) / 2
        val padY = (inputSize - scaledH) / 2

        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        Canvas(letterboxed).apply {
            drawColor(Color.rgb(114, 114, 114)) // Ultralytics' own training-time letterbox pad color
            val scaledBitmap = Bitmap.createScaledBitmap(rotated, scaledW, scaledH, true)
            drawBitmap(scaledBitmap, padX.toFloat(), padY.toFloat(), null)
        }

        val inputBuffer = bitmapToInputBuffer(letterboxed)
        val outputBuffer = allocateOutputBuffer()
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val rawDetections = decodeOutput(outputBuffer)
        val nmsed = nonMaxSuppression(rawDetections)

        // Map boxes from letterboxed 640x640 space back to the rotated bitmap's space.
        return nmsed.map { d ->
            val b = d.box
            d.copy(
                box = RectF(
                    (b.left - padX) / scale,
                    (b.top - padY) / scale,
                    (b.right - padX) / scale,
                    (b.bottom - padY) / scale
                )
            )
        }
    }

    private fun quantize(realValue: Float): Int {
        val q = (realValue / inputQuant.scale + inputQuant.zeroPoint).roundToInt()
        val range = if (inputDataType == DataType.UINT8) 0..255 else -128..127
        return q.coerceIn(range.first, range.last)
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val bytesPerChannel = if (inputDataType == DataType.FLOAT32) 4 else 1
        val buffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * bytesPerChannel)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            if (inputDataType == DataType.FLOAT32) {
                buffer.putFloat(r / 255f)
                buffer.putFloat(g / 255f)
                buffer.putFloat(b / 255f)
            } else {
                buffer.put(quantize(r / 255f).toByte())
                buffer.put(quantize(g / 255f).toByte())
                buffer.put(quantize(b / 255f).toByte())
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun allocateOutputBuffer(): ByteBuffer {
        val bytesPerElement = if (outputDataType == DataType.FLOAT32) 4 else 1
        return ByteBuffer.allocateDirect(numAttrs * numAnchors * bytesPerElement)
            .order(ByteOrder.nativeOrder())
    }

    private fun decodeOutput(outputBuffer: ByteBuffer): List<Detection> {
        val totalElements = numAttrs * numAnchors
        val values = FloatArray(totalElements)

        if (outputDataType == DataType.FLOAT32) {
            outputBuffer.asFloatBuffer().get(values)
        } else {
            for (i in 0 until totalElements) {
                val byteVal = outputBuffer.get(i)
                val raw = if (outputDataType == DataType.UINT8) (byteVal.toInt() and 0xFF) else byteVal.toInt()
                values[i] = (raw - outputQuant.zeroPoint) * outputQuant.scale
            }
        }

        fun valueAt(attrIdx: Int, anchorIdx: Int): Float {
            val idx = if (channelsFirst) attrIdx * numAnchors + anchorIdx else anchorIdx * numAttrs + attrIdx
            return values[idx]
        }

        val detections = mutableListOf<Detection>()
        for (a in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = 0f
            for (c in labels.indices) {
                val score = valueAt(4 + c, a)
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            if (bestClass < 0 || bestScore < confidenceThreshold) continue

            // YOLOv8 export convention: box coords in the model's input pixel space (0..inputSize).
            val cx = valueAt(0, a)
            val cy = valueAt(1, a)
            val w = valueAt(2, a)
            val h = valueAt(3, a)
            detections.add(
                Detection(
                    classId = bestClass,
                    label = labels[bestClass],
                    confidence = bestScore,
                    box = RectF(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
                )
            )
        }
        return detections
    }

    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { it.classId == best.classId && iou(best.box, it.box) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        val union = aArea + bArea - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    fun close() {
        interpreter.close()
        activeDelegate?.close()
    }

    companion object {
        private const val TAG = "ObjectDetectorHelper"
    }
}
