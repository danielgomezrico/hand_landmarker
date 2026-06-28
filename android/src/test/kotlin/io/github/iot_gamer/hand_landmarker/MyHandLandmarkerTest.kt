package io.github.iot_gamer.hand_landmarker

import java.nio.ByteBuffer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Device-free JVM tests for [MyHandLandmarker] correctness properties added in T2.1.
 *
 * Uses the same injection seams as [DelegateFallbackTest]:
 *  - [buildHandLandmarker] returns null so no Android Context, MediaPipe model, or
 *    GPU driver is required.
 *  - [MyHandLandmarker.processSlotFn] overrides the processSlot path so no Bitmap /
 *    BitmapImageBuilder / detectAsync is called — tests are pure JVM.
 *  - [MyHandLandmarker.listener] records errors emitted by drainTask.
 *
 * context=null is safe because the injected builder never dereferences it.
 */
internal class MyHandLandmarkerTest {

    private fun landmarker(): MyHandLandmarker =
        MyHandLandmarker(null) { _, _, _ -> null }

    // ── Re-init after close ─────────────────────────────────────────────────────────────────

    @Test
    fun reinitAfterClose_throwsIllegalStateException() {
        // Proves P2 fix: initialize() after close() gives a clear error instead of
        // silently producing dead frames.
        val lm = landmarker()
        lm.initialize(1, 0.5f, useGpu = false)
        lm.close()
        assertFailsWith<IllegalStateException>("initialize() after close() must throw") {
            lm.initialize(1, 0.5f, useGpu = false)
        }
    }

    // ── processSlot error routing ────────────────────────────────────────────────────────────

    @Test
    fun processSlotThrows_routesErrorToListener() {
        // Proves P0 fix: an exception inside processSlot (simulated via processSlotFn)
        // surfaces to the listener as MEDIAPIPE_ERROR instead of silently killing the
        // drain loop.
        val errors = mutableListOf<Pair<String, String>>()

        val lm = landmarker()
        lm.processSlotFn = { _ -> throw RuntimeException("synthetic slot error") }
        lm.listener = object : HandLandmarkListener {
            override fun onLandmarksDetected(hands: List<FloatArray>) {}
            override fun onError(code: String, message: String) { errors += code to message }
        }
        lm.initialize(1, 0.5f, useGpu = false)

        // Trigger with an empty frame (width=height=0 so no YUV work is done before the
        // override is called; processSlotFn replaces processSlot entirely, so no
        // Bitmap.createBitmap / BitmapImageBuilder / detectAsync are invoked).
        val empty = ByteBuffer.allocate(0)
        lm.processFrame(empty, empty, empty, 0, 0, 0, 0, 0, 0, 1L)

        // Deterministically wait for the worker to drain the frame (close() is now
        // non-blocking, so it can no longer serve as the barrier) before checking 'errors'.
        lm.awaitWorkerIdleForTest(1000)
        lm.close()

        assertEquals(1, errors.size, "exactly one error must be emitted")
        assertEquals("MEDIAPIPE_ERROR", errors[0].first)
        assertTrue(
            errors[0].second.contains("synthetic slot error"),
            "error message must include the exception message, got: ${errors[0].second}"
        )
    }

    // ── After-throw pipeline liveness ────────────────────────────────────────────────────────

    @Test
    fun processSlotThrows_pipelineRemainsLive_secondFrameProcessed() {
        // Proves P0 fix: after one throwing frame, the coordinator's wip is reset so
        // the NEXT frame is also processed (pipeline not permanently dead).
        val processedCount = java.util.concurrent.atomic.AtomicInteger(0)
        var throwOnFirst = true

        val lm = landmarker()
        lm.processSlotFn = { _ ->
            if (throwOnFirst) {
                throwOnFirst = false
                throw RuntimeException("first frame error")
            }
            processedCount.incrementAndGet()
        }
        lm.listener = object : HandLandmarkListener {
            override fun onLandmarksDetected(hands: List<FloatArray>) {}
            override fun onError(code: String, message: String) {}
        }
        lm.initialize(1, 0.5f, useGpu = false)

        val empty = ByteBuffer.allocate(0)
        // First frame — throws inside processSlotFn
        lm.processFrame(empty, empty, empty, 0, 0, 0, 0, 0, 0, 1L)
        // Deterministically drain the first (throwing) frame before enqueuing the second.
        lm.awaitWorkerIdleForTest(1000)
        // Second frame — must be processed now that wip is reset by cancelPending.
        lm.processFrame(empty, empty, empty, 0, 0, 0, 0, 0, 0, 2L)
        lm.awaitWorkerIdleForTest(1000)
        lm.close()

        assertEquals(1, processedCount.get(), "second frame must be processed after first frame threw")
    }
}
