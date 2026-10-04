package com.sightsync.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import android.view.Surface
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Drives an ARCore Session to get per-frame camera images (fed into the existing YOLO
 * detector) plus the Depth API's per-pixel millimeter distance map -- the Phase 3 ask.
 *
 * ARCore's SharedCamera API (the only way to let CameraX keep owning the camera alongside
 * ARCore) explicitly does not support the depth sensor, so when depth is available this
 * controller takes over camera ownership entirely via its own Session rather than coexisting
 * with Phase 2's CameraX pipeline (see MainActivity for the fallback decision). A
 * GLSurfaceView is required -- ARCore needs a valid GL context and an external texture bound
 * via Session.setCameraTextureName() for Session.update() to work at all. onDrawFrame() also
 * blits that texture to the screen (a minimal version of ARCore samples' BackgroundRenderer) so
 * there's a live visual preview -- useful on its own, and the only way to visually confirm the
 * camera orientation/content match what's being fed to the detector.
 */
class ArCoreCameraController(
    private val context: Context,
    private val session: Session,
    private val detector: ObjectDetectorHelper,
    private val onResult: (
        detections: List<ObjectDetectorHelper.Detection>,
        frameWidth: Int,
        frameHeight: Int,
        inferenceMs: Long,
        depthSummary: String
    ) -> Unit
) : GLSurfaceView.Renderer {

    private var textureId = 0
    private var lastTimestampNs = 0L
    private val imageRotationDegrees: Int = computeImageRotationDegrees(context)

    private var shaderProgram = 0
    private var positionAttrib = 0
    private var texCoordAttrib = 0
    private var textureUniform = 0

    // A full-screen quad in OpenGL normalized device coordinates; never changes.
    private val quadPositions: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
            rewind()
        }

    // Texture coords get overwritten every frame by Frame.transformCoordinates2d() below, since
    // the mapping from this quad to the camera texture depends on the display rotation/aspect
    // ratio ARCore is told about via setDisplayGeometry().
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()

    override fun onSurfaceCreated(gl: GL10?, eglConfig: EGLConfig?) {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        session.setCameraTextureName(textureId)

        shaderProgram = buildShaderProgram()
        positionAttrib = GLES20.glGetAttribLocation(shaderProgram, "a_Position")
        texCoordAttrib = GLES20.glGetAttribLocation(shaderProgram, "a_TexCoord")
        textureUniform = GLES20.glGetUniformLocation(shaderProgram, "u_Texture")
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        session.setDisplayGeometry(Surface.ROTATION_0, width, height)
    }

    private fun drawCameraBackground(frame: Frame) {
        frame.transformCoordinates2d(
            Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
            quadPositions,
            Coordinates2d.TEXTURE_NORMALIZED,
            quadTexCoords
        )

        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(shaderProgram)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(textureUniform, 0)

        quadPositions.position(0)
        GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glEnableVertexAttribArray(positionAttrib)

        quadTexCoords.position(0)
        GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(texCoordAttrib)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(positionAttrib)
        GLES20.glDisableVertexAttribArray(texCoordAttrib)
    }

    private fun buildShaderProgram(): Int {
        val vertexShader = compileShader(
            GLES20.GL_VERTEX_SHADER,
            """
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
                gl_Position = a_Position;
                v_TexCoord = a_TexCoord;
            }
            """.trimIndent()
        )
        val fragmentShader = compileShader(
            GLES20.GL_FRAGMENT_SHADER,
            """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 v_TexCoord;
            uniform samplerExternalOES u_Texture;
            void main() {
                gl_FragColor = texture2D(u_Texture, v_TexCoord);
            }
            """.trimIndent()
        )
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        return program
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        return shader
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val frame: Frame = try {
            session.update()
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "Camera not available in ARCore session", e)
            return
        }

        drawCameraBackground(frame)

        // GLSurfaceView calls onDrawFrame() at display refresh rate; skip re-processing the
        // same camera frame if ARCore hasn't delivered a new one since last time.
        val timestampNs = frame.timestamp
        if (timestampNs == lastTimestampNs) return
        lastTimestampNs = timestampNs

        val depthSummary = readDepthSummary(frame)

        val t0 = System.nanoTime()
        var resultDetections: List<ObjectDetectorHelper.Detection> = emptyList()
        var frameWidth = 0
        var frameHeight = 0
        try {
            val cameraImage = frame.acquireCameraImage()
            try {
                frameWidth = cameraImage.width
                frameHeight = cameraImage.height
                val bitmap = imageToBitmap(cameraImage, imageRotationDegrees)
                resultDetections = detector.detect(bitmap, 0)
            } finally {
                cameraImage.close()
            }
        } catch (e: NotYetAvailableException) {
            // Normal on the first few frames while ARCore is still warming up.
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed for this ARCore frame", e)
        }
        val inferenceMs = (System.nanoTime() - t0) / 1_000_000

        onResult(resultDetections, frameWidth, frameHeight, inferenceMs, depthSummary)
    }

    private fun readDepthSummary(frame: Frame): String {
        return try {
            frame.acquireDepthImage16Bits().use { depthImage ->
                summarizeDepthImage(depthImage)
            }
        } catch (e: NotYetAvailableException) {
            "depth n/a (warming up)"
        } catch (e: Exception) {
            Log.w(TAG, "Depth image unavailable this frame", e)
            "depth n/a"
        }
    }

    private fun summarizeDepthImage(depthImage: Image): String {
        val plane = depthImage.planes[0]
        val buffer = plane.buffer.order(ByteOrder.nativeOrder())
        val width = depthImage.width
        val height = depthImage.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        var sum = 0L
        var count = 0
        var min = Int.MAX_VALUE
        var max = 0

        // Read the center pixel directly -- the strided scan below steps by 4 and, depending on
        // width/height, can skip the exact center row/column entirely (e.g. height=90 means
        // centerY=45 is never a multiple of 4), silently leaving it stuck at a default value.
        val centerIndex = (height / 2) * rowStride + (width / 2) * pixelStride
        val centerMm = buffer.getShort(centerIndex).toInt() and 0xFFFF

        // Sampling every 4th pixel is plenty for a summary stat and much cheaper than
        // reading every pixel every frame.
        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val index = y * rowStride + x * pixelStride
                val mm = buffer.getShort(index).toInt() and 0xFFFF
                if (mm > 0) { // 0 = "no valid depth estimate at this pixel"
                    sum += mm
                    count++
                    if (mm < min) min = mm
                    if (mm > max) max = mm
                }
                x += 4
            }
            y += 4
        }

        return if (count == 0) {
            "depth: no valid pixels (${width}x${height})"
        } else {
            "depth: center=${centerMm}mm avg=${sum / count}mm min=${min}mm max=${max}mm (${width}x${height})"
        }
    }

    private fun imageToBitmap(image: Image, rotationDegrees: Int): Bitmap {
        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        return if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }

    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val nv21 = ByteArray(width * height + 2 * (width / 2) * (height / 2))

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        var pos = 0
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        for (row in 0 until height) {
            if (yPixelStride == 1) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(nv21, pos, width)
                pos += width
            } else {
                val rowStart = row * yRowStride
                for (col in 0 until width) {
                    nv21[pos++] = yBuffer.get(rowStart + col * yPixelStride)
                }
            }
        }

        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        val chromaWidth = width / 2
        val chromaHeight = height / 2
        for (row in 0 until chromaHeight) {
            val vRowStart = row * vRowStride
            val uRowStart = row * uRowStride
            for (col in 0 until chromaWidth) {
                nv21[pos++] = vBuffer.get(vRowStart + col * vPixelStride)
                nv21[pos++] = uBuffer.get(uRowStart + col * uPixelStride)
            }
        }
        return nv21
    }

    companion object {
        private const val TAG = "ArCoreCameraController"

        private fun computeImageRotationDegrees(context: Context): Int {
            return try {
                val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                    cameraManager.getCameraCharacteristics(id)
                        .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
                } ?: cameraManager.cameraIdList.firstOrNull() ?: return 90

                val sensorOrientation = cameraManager.getCameraCharacteristics(cameraId)
                    .get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

                // MainActivity is portrait-locked (see AndroidManifest), so device rotation
                // is always Surface.ROTATION_0 here -- this reduces to the sensor
                // orientation itself, matching the rotationDegrees CameraX reported in
                // Phase 1/2 for the same physical camera.
                sensorOrientation
            } catch (e: Exception) {
                Log.w(TAG, "Could not determine camera sensor orientation, defaulting to 90", e)
                90
            }
        }
    }
}
