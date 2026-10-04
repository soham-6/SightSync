package com.sightsync.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Debug overlay drawing the latest frame's detection boxes + labels on top of the camera
 * preview. Dev tooling for Phase 2 verification -- the final product is audio-only, this view
 * doesn't commit to any final UI design. The scaling here is an approximation (doesn't account
 * for PreviewView's internal crop vs the analyzer's own frame crop) -- fine for "does detection
 * roughly work," not meant to be pixel-perfect AR alignment.
 */
class DetectionOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var detections: List<ObjectDetectorHelper.Detection> = emptyList()
    private var sourceWidth = 1
    private var sourceHeight = 1

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.GREEN
    }
    private val textPaint = Paint().apply {
        color = Color.GREEN
        textSize = 42f
        isAntiAlias = true
    }
    private val textBackgroundPaint = Paint().apply {
        color = Color.BLACK
        alpha = 160
    }

    fun setDetections(detections: List<ObjectDetectorHelper.Detection>, sourceWidth: Int, sourceHeight: Int) {
        this.detections = detections
        this.sourceWidth = sourceWidth
        this.sourceHeight = sourceHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (sourceWidth <= 0 || sourceHeight <= 0) return

        val scaleX = width.toFloat() / sourceWidth
        val scaleY = height.toFloat() / sourceHeight

        for (d in detections) {
            val r = RectF(
                d.box.left * scaleX,
                d.box.top * scaleY,
                d.box.right * scaleX,
                d.box.bottom * scaleY
            )
            canvas.drawRect(r, boxPaint)

            val label = "${d.label} ${"%.2f".format(d.confidence)}"
            val textWidth = textPaint.measureText(label)
            canvas.drawRect(r.left, r.top - 48f, r.left + textWidth + 8f, r.top, textBackgroundPaint)
            canvas.drawText(label, r.left + 4f, r.top - 10f, textPaint)
        }
    }
}
