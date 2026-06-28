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
import java.util.concurrent.Executors
import java.util.concurrent.CountDownLatch
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

// K4 frozen interface — wire payload: List<FloatArray> (x,y,z interleaved, one per hand)
interface HandLandmarkListener {
    fun onLandmarksDetected(hands: List<FloatArray>)
    fun onError(code: String, message: String)
}

class MyHandLandmarker @JvmOverloads constructor(
    // Nullable for JVM-only tests: the injected builder never uses context, so null is safe.
    private val context: Context?,
    // Injectable seam — overridable for JVM-only tests (mirrors TimestampGate extraction).
    // Returns HandLandmarker? so tests can return null without instantiating the final class.
    // Null sentinel means use the production MediaPipe path set up in init{}.
    internal var buildHandLandmarker: ((numHands: Int, confidence: Float, delegate: Delegate) -> HandLandmarker?)? = null
) {
    // @Volatile: read on worker thread, nulled on close() called from main thread.
    @Volatile private var handLandmarker: HandLandmarker? = null
    // @Volatile: written on any thread (init or close); read on MediaPipe callback thread.
    @Volatile var listener: HandLandmarkListener? = null

    // Injectable seam — tests override this to avoid Android/MediaPipe calls in processSlot.
    // When null (production path), processSlot is used directly.
    internal var processSlotFn: ((FrameSlot) -> Unit)? = null
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

    // T2.1 — latest-frame-wins worker-thread offload.
    // [coordinator] is touched by both producer and worker threads — uses concurrent types.
    private val coordinator = FrameDrainCoordinator<FrameSlot>()
    private val worker = Executors.newSingleThreadExecutor()
    private val drainTask = Runnable {
        try {
            // In tests, processSlotFn overrides processSlot to avoid Android/MediaPipe.
            coordinator.drain { slot -> (processSlotFn ?: ::processSlot)(slot) }
        } catch (e: Throwable) {
            listener?.onError("MEDIAPIPE_ERROR", e.message ?: "unknown error in drain")
            // drain's wip decrement only lands at 0 when NO frame arrived during the throwing
            // process; a frame enqueued concurrently leaves wip>=1 and would permanently strand
            // the pipeline (no future enqueue schedules a drain). cancelPending() unconditionally
            // resets wip=0 and recycles any stranded slot, so the next camera frame recovers —
            // dropping only the in-flight frame(s) during the error window.
            coordinator.cancelPending()
        }
    }
    // @Volatile: written by close() on main thread, checked on producer (camera) thread.
    @Volatile private var closed = false

    // Worker-confined fields — touched ONLY by the single worker thread; no sync needed.
    // Single worker + latest-wins intake gate ⇒ detectAsync always sees strictly increasing
    // timestamps (MediaPipe LIVE_STREAM requirement).
    private var argbArray: IntArray = IntArray(0)
    private var poolWidth: Int = 0
    private var poolHeight: Int = 0
    // Int.MIN_VALUE sentinel: a valid rotation (0/90/180/270) never matches this on first frame,
    // ensuring cachedOptions is populated before the first detectAsync call.
    private var cachedRotation: Int = Int.MIN_VALUE
    private var cachedOptions: ImageProcessingOptions? = null

    // T2 — dropped-frame debug counter (camera-thread-confined)
    private var droppedFrameCount: Long = 0L

    fun initialize(
        numHands: Int,
        minHandDetectionConfidence: Float,
        useGpu: Boolean
    ) {
        check(!closed) { "MyHandLandmarker has been closed; create a new instance" }
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
        // T2.1 — fast-path: engine closing or not yet initialized
        if (closed) return
        // processSlotFn is non-null only in tests; in production, handLandmarker == null
        // means the engine is not yet initialized — skip frame to avoid wasted work.
        if (handLandmarker == null && processSlotFn == null) return

        // T2 — CAS monotonic guard: drop OOO/duplicate frames before any YUV work.
        // Decision routed through the unit-tested TimestampGate seam (tested == shipped).
        val prev = lastTimestampMs.get()
        if (!TimestampGate.shouldProcess(timestampMs, prev) || !lastTimestampMs.compareAndSet(prev, timestampMs)) {
            droppedFrameCount++
            Log.d("MyHandLandmarker", "Dropped OOO frame ts=$timestampMs prev=$prev (total dropped: $droppedFrameCount)")
            return
        }

        // T2.1 — Acquire a slot from the free pool (or allocate a new one).
        // The plane copies below are the necessary ownership transfer: Dart-side JNI ByteBuffers
        // are reused/overwritten on the next frame, so we must copy before returning.
        val slot = coordinator.pollFree() ?: FrameSlot()

        // Resize slot arrays only if needed — free-pool slots may already be large enough.
        val pixels = width * height
        val uCap = uBuffer.capacity()
        val vCap = vBuffer.capacity()
        if (slot.yArr.size < pixels) slot.yArr = ByteArray(pixels)
        if (slot.uArr.size < uCap) slot.uArr = ByteArray(uCap)
        if (slot.vArr.size < vCap) slot.vArr = ByteArray(vCap)

        // T7 — fill Y plane row-by-row into slot's tight array (strips yRowStride padding)
        var yIdx = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(slot.yArr, yIdx, width)
            yIdx += width
        }

        // T7 — fill U/V planes wholesale into slot arrays (capacity-sized)
        uBuffer.position(0); uBuffer.get(slot.uArr, 0, uCap)
        vBuffer.position(0); vBuffer.get(slot.vArr, 0, vCap)

        // Store frame metadata on the slot
        slot.width = width
        slot.height = height
        slot.uvRowStride = uvRowStride
        slot.uvPixelStride = uvPixelStride
        slot.rotation = rotation
        slot.timestampMs = timestampMs

        // Latest-frame-wins: enqueue displaces any unprocessed prior slot to the free pool.
        // Returns true only when wip transitions 0→1 (no drain currently running).
        if (coordinator.enqueue(slot)) {
            try {
                worker.execute(drainTask)
            } catch (e: RejectedExecutionException) {
                // close() raced with this execute() and won — the executor is shut down.
                // The just-enqueued slot is stranded in pending; return it to the free
                // pool so the coordinator is left leak-free.
                coordinator.cancelPending()
            }
        }
    }

    /**
     * Runs ONLY on the worker thread (via [drainTask]/[coordinator.drain]).
     * Worker-confined fields (argbArray, poolWidth/Height, cachedRotation/Options)
     * need no synchronization.
     *
     * Single worker + latest-wins intake gate ⇒ detectAsync always sees strictly
     * increasing timestamps (MediaPipe LIVE_STREAM requirement preserved).
     */
    private fun processSlot(slot: FrameSlot) {
        // Reallocate worker ARGB buffer only when frame dimensions change.
        // Routing through the testable seam so the sizing policy can be unit-tested
        // without a Bitmap or Android dependency.
        argbArray = ensureArgbCapacity(argbArray, slot.width, slot.height, poolWidth, poolHeight)
        poolWidth = slot.width
        poolHeight = slot.height

        // T7 — YUV -> ARGB conversion into worker-owned buffer
        YuvConverter.yuv420ToArgb(
            argbArray, slot.yArr, slot.uArr, slot.vArr,
            slot.width, slot.height,
            yStride = slot.width, uvRowStride = slot.uvRowStride, uvPixelStride = slot.uvPixelStride
        )

        val bitmap = Bitmap.createBitmap(argbArray, slot.width, slot.height, Bitmap.Config.ARGB_8888)
        val mpImage = BitmapImageBuilder(bitmap).build()

        // T7 — cache ImageProcessingOptions per rotation (worker-confined; no sync needed)
        if (slot.rotation != cachedRotation) {
            cachedOptions = ImageProcessingOptions.builder().setRotationDegrees(slot.rotation).build()
            cachedRotation = slot.rotation
        }

        val lm = handLandmarker
        try {
            lm?.detectAsync(mpImage, cachedOptions!!, slot.timestampMs)
        } finally {
            // detectAsync copies the bitmap's pixels into a native ImageFrame synchronously on
            // this thread before returning, so the MPImage can be closed now — even if
            // detectAsync throws. The bitmap is intentionally NOT recycled — GC reclaims it,
            // matching the proven pre-refactor path (62d0241). Recycling here risked
            // corrupting in-flight pixels on some paths.
            mpImage.close()
        }
    }

    /**
     * Test-only: blocks until the single-thread worker has drained all work submitted
     * before this call. Because the executor is single-threaded and FIFO, a barrier task
     * runs only after every previously-submitted drainTask has completed. Replaces the
     * old habit of using [close] as a drain barrier (close() is now non-blocking).
     */
    internal fun awaitWorkerIdleForTest(timeoutMs: Long) {
        val latch = CountDownLatch(1)
        try {
            worker.execute { latch.countDown() }
        } catch (e: RejectedExecutionException) {
            return // worker already shut down — nothing in flight
        }
        latch.await(timeoutMs, TimeUnit.MILLISECONDS)
    }

    /**
     * Shuts down the worker and releases the native MediaPipe engine.
     *
     * Returns to the caller IMMEDIATELY: the bounded worker drain and the native
     * engine teardown are handed to a detached thread, so a caller on the platform/
     * main thread (e.g. Dart dispose()) never blocks — no ANR risk, no Dart-isolate
     * plumbing required. Idempotent; safe to call from any thread.
     *
     * Thread safety: [handLandmarker] is nulled synchronously BEFORE the teardown
     * thread runs, so any worker thread past the null-check operates on a local
     * capture and cannot race [lm.close()]. Correctness does NOT depend on the wait.
     */
    // T3 — lifecycle: shut down the worker before closing the native engine so detectAsync
    // cannot be called after handLandmarker.close().
    fun close() {
        if (closed) return // idempotent — never spawn a second teardown thread
        closed = true
        worker.shutdown()
        // Capture and null the engine synchronously so producers/workers see null
        // immediately; the teardown thread owns `lm` for the rest of its life.
        val lm = handLandmarker
        handLandmarker = null
        // Offload the blocking drain + native teardown so the calling (main) thread
        // returns at once. The lambda captures `worker` (a field) and `lm`, which keeps
        // this instance and the engine alive in the JVM even after Dart release()s its ref.
        Thread {
            try {
                // close-contract (MediaPipe Tasks-Vision, LIVE_STREAM): HandLandmarker.close() ->
                // TaskRunner.close() runs closeAllPacketSources() then waitUntilGraphDone() (blocks
                // until every packet propagates and every calculator Close()s) BEFORE tearDown() frees
                // native state. LIVE_STREAM result listeners fire synchronously on graph worker threads
                // during execution, so they are fully drained before close() returns — no
                // callback-after-close / use-after-free on the normal path. close() is NOT thread-safe
                // vs detectAsync(); we stop the worker (shutdown + awaitTermination) and null-guard the
                // engine above BEFORE close() so no detectAsync can race the teardown.
                // Ref: TaskRunner.java close(); CalculatorGraph::WaitUntilDone();
                // ai.google.dev/edge/mediapipe/framework/getting_started/troubleshooting
                worker.awaitTermination(250, TimeUnit.MILLISECONDS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            }
            lm?.close()
        }.apply { name = "hand-landmarker-teardown"; isDaemon = true }.start()
    }
}

/**
 * Returns the ARGB buffer to use for the next frame, potentially reusing [current].
 *
 * This function is extracted from [MyHandLandmarker.processSlot] so the buffer-sizing
 * policy can be unit-tested on the JVM without touching Bitmap or Android classes.
 *
 * Current (dim-based) policy: reallocate whenever width or height changes.
 * The production fix (high-water-mark) is applied in a follow-up commit after the
 * RED test proves this dim-based version reallocates on same-pixel-count dim changes.
 */
internal fun ensureArgbCapacity(
    current: IntArray,
    width: Int,
    height: Int,
    prevWidth: Int,
    prevHeight: Int
): IntArray = if (width != prevWidth || height != prevHeight) IntArray(width * height) else current

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
