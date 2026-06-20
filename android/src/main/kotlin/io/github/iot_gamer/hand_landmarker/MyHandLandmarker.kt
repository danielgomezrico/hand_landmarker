package io.github.iot_gamer.hand_landmarker

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicLong

// K4 frozen interface — wire payload: List<FloatArray> (x,y,z interleaved, one per hand)
interface HandLandmarkListener {
    fun onLandmarksDetected(hands: List<FloatArray>)
    fun onError(code: String, message: String)
}

class MyHandLandmarker(
    // Nullable for JVM-only tests: the injected builder never uses context, so null is safe.
    private val context: Context?,
    // Injectable seam — overridable for JVM-only tests (mirrors TimestampGate extraction).
    // Returns HandLandmarker? so tests can return null without instantiating the final class.
    // Null sentinel means use the production MediaPipe path set up in init{}.
    internal var buildHandLandmarker: ((numHands: Int, confidence: Float, delegate: Delegate) -> HandLandmarker?)? = null
) {
    private var handLandmarker: HandLandmarker? = null
    var listener: HandLandmarkListener? = null
    var activeDelegate: Delegate = Delegate.CPU

    init {
        if (buildHandLandmarker == null) {
            buildHandLandmarker = { numHands, confidence, delegate ->
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .setDelegate(delegate)
                    .build()
                val options = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setNumHands(numHands)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setMinHandDetectionConfidence(confidence)
                    .setResultListener { result, _ ->
                        // K4 — emit encoded binary payload; never call plugin static
                        val hands = if (result == null || result.landmarks().isEmpty()) {
                            emptyList()
                        } else {
                            LandmarksEncoder.encode(result.landmarks())
                        }
                        listener?.onLandmarksDetected(hands)
                    }
                    .setErrorListener { error ->
                        // K4 — route errors through the listener interface
                        listener?.onError("MEDIAPIPE_ERROR", error.message ?: "unknown")
                    }
                    .build()
                HandLandmarker.createFromOptions(context, options)
            }
        }
    }


    // T2 — monotonic CAS guard (accumulateAndGet-max is BANNED per W1)
    private val lastTimestampMs = AtomicLong(-1L)

    // T7 — session-scoped array pool (camera-thread-confined; realloc only on dim/capacity change)
    private var yArr: ByteArray = ByteArray(0)
    private var uArr: ByteArray = ByteArray(0)
    private var vArr: ByteArray = ByteArray(0)
    private var argbArray: IntArray = IntArray(0)
    private var poolWidth: Int = 0
    private var poolHeight: Int = 0

    // T7 — cached ImageProcessingOptions per rotation
    private var cachedRotation: Int = Int.MIN_VALUE
    private var cachedOptions: ImageProcessingOptions? = null

    // T2 — dropped-frame debug counter (Android-bound; kept out of pure TimestampGate seam)
    private var droppedFrameCount: Long = 0L

    fun initialize(
        numHands: Int,
        minHandDetectionConfidence: Float,
        useGpu: Boolean
    ) {
        val builder = buildHandLandmarker!!
        // Close any prior engine before reassigning so a re-initialize never leaks the native instance.
        handLandmarker?.close()
        if (useGpu) {
            try {
                handLandmarker = builder(numHands, minHandDetectionConfidence, Delegate.GPU)
                activeDelegate = Delegate.GPU
            } catch (e: RuntimeException) {
                // GPU delegate unavailable (e.g. missing OpenCL driver) — fall back to CPU.
                Log.w("MyHandLandmarker", "GPU delegate failed, falling back to CPU: ${e.message}")
                handLandmarker = builder(numHands, minHandDetectionConfidence, Delegate.CPU)
                activeDelegate = Delegate.CPU
            }
        } else {
            handLandmarker = builder(numHands, minHandDetectionConfidence, Delegate.CPU)
            activeDelegate = Delegate.CPU
        }

        // Adopt the engine plugin's EventChannel sink as our result listener.
        // Dart builds this object via JNI, so it has no handle to the FlutterPlugin;
        // the plugin published itself at engine attach. Guarded so an explicitly
        // injected listener (unit tests) is never overwritten.
        if (listener == null) {
            listener = HandLandmarkerPlugin.activeListener
        }
    }

    fun getActiveDelegate(): String = activeDelegate.name

    fun processFrame(
        yBuffer: ByteBuffer, uBuffer: ByteBuffer, vBuffer: ByteBuffer,
        width: Int, height: Int, yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotation: Int, timestampMs: Long
    ) {
        if (handLandmarker == null) return

        // T2 — CAS monotonic guard: drop OOO/duplicate frames before YUV work.
        // Decision routed through the unit-tested TimestampGate seam (tested == shipped).
        val prev = lastTimestampMs.get()
        if (!TimestampGate.shouldProcess(timestampMs, prev) || !lastTimestampMs.compareAndSet(prev, timestampMs)) {
            droppedFrameCount++
            Log.d("MyHandLandmarker", "Dropped OOO frame ts=$timestampMs prev=$prev (total dropped: $droppedFrameCount)")
            return
        }

        // T7 — ensure array pool is sized for this frame's dimensions
        ensurePools(width, height, uBuffer.capacity(), vBuffer.capacity())

        // T7 — fill Y plane row-by-row into pooled tight array
        var yIdx = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(yArr, yIdx, width)
            yIdx += width
        }

        // T7 — fill U/V planes wholesale into pooled arrays (capacity-sized)
        uBuffer.position(0); uBuffer.get(uArr, 0, uBuffer.capacity())
        vBuffer.position(0); vBuffer.get(vArr, 0, vBuffer.capacity())

        // T7 — run YUV -> ARGB into pooled output array
        YuvConverter.yuv420ToArgb(
            argbArray, yArr, uArr, vArr, width, height,
            yStride = width, uvRowStride = uvRowStride, uvPixelStride = uvPixelStride
        )

        val bitmap = Bitmap.createBitmap(argbArray, width, height, Bitmap.Config.ARGB_8888)
        val mpImage = BitmapImageBuilder(bitmap).build()

        // T7 — cache ImageProcessingOptions per rotation
        if (rotation != cachedRotation) {
            cachedOptions = ImageProcessingOptions.builder().setRotationDegrees(rotation).build()
            cachedRotation = rotation
        }

        handLandmarker?.detectAsync(mpImage, cachedOptions!!, timestampMs)

        // detectAsync copies the bitmap's pixels into a native ImageFrame synchronously on
        // this thread before returning, so the MPImage can be closed now. The bitmap is
        // intentionally NOT recycled — GC reclaims it, matching the proven pre-refactor
        // path (62d0241). Recycling here risked corrupting in-flight pixels on some paths.
        mpImage.close()
    }

    // T3 — lifecycle: close native resources
    fun close() {
        handLandmarker?.close()
        handLandmarker = null
    }

    /**
     * T7 — ensures pooled arrays are sized for the given frame dimensions.
     * Reallocates only when dimensions or U/V buffer capacities change.
     * Sizes U/V arrays to their full buffer capacity (handles semi-planar uvPixelStride=2).
     * Pure logic: no Android imports — could be extracted to a testable seam if needed.
     */
    private fun ensurePools(width: Int, height: Int, uCap: Int, vCap: Int) {
        val pixels = width * height
        if (width != poolWidth || height != poolHeight) {
            yArr = ByteArray(pixels)
            argbArray = IntArray(pixels)
            poolWidth = width
            poolHeight = height
        }
        if (uArr.size < uCap) uArr = ByteArray(uCap)
        if (vArr.size < vCap) vArr = ByteArray(vCap)
    }
}

/**
 * Pure (Android-free, JVM-unit-testable) YUV_420_888 -> ARGB pixel math.
 *
 * Extracted from the detection hot path so the fixed-point integer color
 * conversion can be regression-tested without a device, a Bitmap, or MediaPipe.
 */
internal object YuvConverter {

    /**
     * Converts the [y]/[u]/[v] planes into 0xFF-alpha ARGB ints written to [out]
     * (which must have length >= width*height), using fixed-point integer BT.601
     * (JFIF full-range) coefficients (<<10):
     *   R = Y + 1.402·(V-128)
     *   G = Y - 0.344136·(U-128) - 0.714136·(V-128)
     *   B = Y + 1.772·(U-128)
     *
     * The Y plane is indexed with [yStride] (pass `width` for a tight buffer);
     * U/V are indexed with [uvRowStride]/[uvPixelStride], covering both planar
     * (pixelStride=1) and semi-planar (pixelStride=2) layouts.
     */
    fun yuv420ToArgb(
        out: IntArray,
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        width: Int,
        height: Int,
        yStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int
    ) {
        for (j in 0 until height) {
            val yRow = j * yStride
            val outRow = j * width
            val uvRow = (j shr 1) * uvRowStride
            for (i in 0 until width) {
                val yy = (y[yRow + i].toInt() and 0xFF)
                val uvCol = (i shr 1) * uvPixelStride
                val uu = (u[uvRow + uvCol].toInt() and 0xFF) - 128
                val vv = (v[uvRow + uvCol].toInt() and 0xFF) - 128
                // +512 biases the >>10 to round-to-nearest (not truncate), halving
                // the error vs the exact real-valued conversion and removing the
                // systematic downward (darkening) bias. shr floors, so (x+512)>>10
                // is correct round-half-up for negative terms too.
                var r = yy + ((1436 * vv + 512) shr 10)
                var g = yy - ((352 * uu + 731 * vv + 512) shr 10)
                var b = yy + ((1815 * uu + 512) shr 10)
                r = if (r < 0) 0 else if (r > 255) 255 else r
                g = if (g < 0) 0 else if (g > 255) 255 else g
                b = if (b < 0) 0 else if (b > 255) 255 else b
                out[outRow + i] = (0xFF000000.toInt()) or (r shl 16) or (g shl 8) or b
            }
        }
    }
}
