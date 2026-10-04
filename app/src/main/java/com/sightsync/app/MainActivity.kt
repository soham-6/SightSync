package com.sightsync.app

import android.Manifest
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var arGlSurfaceView: GLSurfaceView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var detector: ObjectDetectorHelper

    private var arCoreSession: Session? = null
    private var arCoreController: ArCoreCameraController? = null
    private var usingArCore = false
    private var cameraPipelineStarted = false
    private var arCoreInstallRequested = false

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            // Nothing to do here on grant: onResume() runs right after this callback returns
            // and will start the pipeline. Starting it from both places would race -- if ARCore
            // isn't installed yet, the INSTALL_REQUESTED branch doesn't set cameraPipelineStarted
            // (Play Store takes over the screen), so a second concurrent call here would re-enter
            // tryStartArCore() before the user ever saw ARCore's own install prompt.
            if (!granted) Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        arGlSurfaceView = findViewById(R.id.arGlSurfaceView)
        overlayView = findViewById(R.id.detectionOverlay)
        cameraExecutor = Executors.newSingleThreadExecutor()
        detector = ObjectDetectorHelper(this)

        arGlSurfaceView.setEGLContextClientVersion(2)
        arGlSurfaceView.preserveEGLContextOnPause = true

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
        // If permission is already granted, onResume() (always called right after onCreate())
        // starts the pipeline -- see the comment on requestPermissionLauncher for why this isn't
        // also triggered from here.
    }

    /**
     * Phase 3 entry point: try ARCore + Depth API first. Falls back to Phase 2's CameraX +
     * YOLO pipeline (no depth) whenever ARCore or its Depth API isn't available -- which, per
     * Google's own developer docs, is normal on roughly 1 in 8 active devices, and is also
     * exactly what happens on every Android emulator (ARCore Session creation always fails
     * there with UnavailableDeviceNotCompatibleException; there's no way to test the depth
     * path itself without a real device).
     */
    private fun startCameraPipeline() {
        if (cameraPipelineStarted) return
        tryStartArCore()
    }

    private fun tryStartArCore() {
        try {
            // Per ArCoreApk's documented contract (matching Google's own HelloAR sample): pass
            // true on the first call to allow ARCore's install/update prompt to show, then false
            // on the retry from onResume() after returning from that flow -- at that point
            // INSTALL_REQUESTED/no-op means the user declined, surfaced as
            // UnavailableUserDeclinedInstallationException below. Passing this backwards (as an
            // earlier version of this code did) races the prompt against onResume()'s retry call
            // in the same lifecycle pass and can skip straight past it into the Play Store flow.
            when (ArCoreApk.getInstance().requestInstall(this, !arCoreInstallRequested)) {
                ArCoreApk.InstallStatus.INSTALLED -> {
                    val session = Session(this)
                    val config = session.config
                    if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                        config.depthMode = Config.DepthMode.AUTOMATIC
                        session.configure(config)
                        startArCorePipeline(session)
                    } else {
                        Log.w(TAG, "Device doesn't support ARCore's Depth API; falling back to CameraX (no depth)")
                        session.close()
                        startCameraXFallback()
                    }
                }
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    // ARCore's own install/update flow (and possibly Play Store) takes over the
                    // screen; we retry from onResume() when control comes back to us.
                    arCoreInstallRequested = true
                }
            }
        } catch (e: UnavailableUserDeclinedInstallationException) {
            Log.w(TAG, "User declined ARCore install; falling back to CameraX (no depth)", e)
            startCameraXFallback()
        } catch (e: UnavailableDeviceNotCompatibleException) {
            Log.w(TAG, "Device not ARCore-compatible (expected on emulators); falling back to CameraX (no depth)", e)
            startCameraXFallback()
        } catch (e: UnavailableApkTooOldException) {
            Log.w(TAG, "ARCore APK too old; falling back to CameraX (no depth)", e)
            startCameraXFallback()
        } catch (e: UnavailableSdkTooOldException) {
            Log.w(TAG, "ARCore SDK too old; falling back to CameraX (no depth)", e)
            startCameraXFallback()
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error starting ARCore; falling back to CameraX (no depth)", e)
            startCameraXFallback()
        }
    }

    private fun startArCorePipeline(session: Session) {
        cameraPipelineStarted = true
        usingArCore = true
        arCoreSession = session
        Log.i(TAG, "Using ARCore Depth API pipeline")

        val controller = ArCoreCameraController(this, session, detector) { detections, w, h, inferenceMs, depthSummary ->
            Log.d(
                TAG,
                "Detections=${detections.size} inferenceMs=$inferenceMs $depthSummary " +
                    detections.joinToString { "${it.label}:${"%.2f".format(it.confidence)}" }
            )
            runOnUiThread { overlayView.setDetections(detections, w, h) }
        }
        arCoreController = controller

        previewView.visibility = View.GONE
        arGlSurfaceView.visibility = View.VISIBLE
        arGlSurfaceView.setRenderer(controller)
        arGlSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        arGlSurfaceView.onResume()

        try {
            session.resume()
        } catch (e: Exception) {
            Log.e(TAG, "ARCore session.resume() failed; falling back to CameraX (no depth)", e)
            usingArCore = false
            arGlSurfaceView.visibility = View.GONE
            previewView.visibility = View.VISIBLE
            session.close()
            arCoreSession = null
            startCameraXFallback()
        }
    }

    private fun startCameraXFallback() {
        cameraPipelineStarted = true
        usingArCore = false
        Log.i(TAG, "Using CameraX + YOLO pipeline (no depth)")
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FrameAnalyzer(detector) { detections, w, h, inferenceMs ->
                        Log.d(
                            TAG,
                            "Detections=${detections.size} inferenceMs=$inferenceMs " +
                                detections.joinToString { "${it.label}:${"%.2f".format(it.confidence)}" }
                        )
                        runOnUiThread { overlayView.setDetections(detections, w, h) }
                    })
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onResume() {
        super.onResume()
        if (!cameraPipelineStarted &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        ) {
            // Retry in case an ARCore install completed via Play Store while we were paused.
            tryStartArCore()
        }
        if (usingArCore) {
            arGlSurfaceView.onResume()
            try {
                arCoreSession?.resume()
            } catch (e: Exception) {
                Log.e(TAG, "ARCore session.resume() failed on onResume", e)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (usingArCore) {
            arCoreSession?.pause()
            arGlSurfaceView.onPause()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detector.close()
        arCoreSession?.close()
    }

    private class FrameAnalyzer(
        private val detector: ObjectDetectorHelper,
        private val onResult: (List<ObjectDetectorHelper.Detection>, Int, Int, Long) -> Unit
    ) : ImageAnalysis.Analyzer {
        private var lastTimestampNs = 0L
        private var frameCount = 0

        override fun analyze(imageProxy: ImageProxy) {
            try {
                val width = imageProxy.width
                val height = imageProxy.height
                val nowNs = System.nanoTime()
                frameCount++
                val fpsText = if (lastTimestampNs != 0L) {
                    val deltaMs = (nowNs - lastTimestampNs) / 1_000_000.0
                    if (deltaMs > 0) String.format("%.1f", 1000.0 / deltaMs) else "n/a"
                } else "n/a (first frame)"
                lastTimestampNs = nowNs
                Log.d(
                    TAG,
                    "Frame #$frameCount ${width}x$height ts=${imageProxy.imageInfo.timestamp} instFps=$fpsText"
                )

                val t0 = System.nanoTime()
                val detections = try {
                    val bitmap = imageProxy.toBitmap()
                    detector.detect(bitmap, imageProxy.imageInfo.rotationDegrees)
                } catch (e: Exception) {
                    Log.e(TAG, "Detection failed for this frame", e)
                    emptyList()
                }
                val inferenceMs = (System.nanoTime() - t0) / 1_000_000
                onResult(detections, width, height, inferenceMs)
            } finally {
                imageProxy.close()
            }
        }
    }

    companion object {
        private const val TAG = "SightSyncCamera"
    }
}
