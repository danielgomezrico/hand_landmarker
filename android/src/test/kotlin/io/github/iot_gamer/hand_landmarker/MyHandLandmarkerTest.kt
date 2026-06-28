package io.github.iot_gamer.hand_landmarker

import java.nio.ByteBuffer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertSame
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

    // ── R2-2: processFrame after close does no work ────────────────────────────────────────

    /**
     * R2-2 — GUARD: after close(), processFrame must return immediately without acquiring
     * a slot or calling processSlotFn.
     *
     * Mutation proof: inverting `if (closed) return` to `if (!closed) return` causes
     * processFrame to skip the pre-close frame, so processedCount remains 0 after frame 1,
     * failing the first assertEquals.
     */
    @Test
    fun processFrame_afterClose_noSlotAcquiredNoProcessingDone() {
        val processedCount = java.util.concurrent.atomic.AtomicInteger(0)
        val lm = MyHandLandmarker(null) { _, _, _ -> null }
        lm.processSlotFn = { _ -> processedCount.incrementAndGet() }

        // Frame 1: before close — must be processed normally
        lm.processFrame(
            ByteBuffer.wrap(ByteArray(1)), ByteBuffer.wrap(ByteArray(1)), ByteBuffer.wrap(ByteArray(1)),
            1, 1, 1, 1, 1, 0, 1L
        )
        lm.awaitWorkerIdleForTest(500)
        assertEquals(1, processedCount.get(), "frame before close must be processed")

        // Close — sets closed=true and shuts down the executor
        lm.close()

        // Frame 2: after close — processFrame must return immediately (closed guard)
        lm.processFrame(
            ByteBuffer.wrap(ByteArray(1)), ByteBuffer.wrap(ByteArray(1)), ByteBuffer.wrap(ByteArray(1)),
            1, 1, 1, 1, 1, 0, 2L
        )
        Thread.sleep(100) // allow any erroneous async path time to surface
        assertEquals(1, processedCount.get(), "processFrame after close must NOT increment processedCount")
    }

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

    // ── R2-1: limited/sliced ByteBuffer — no BufferUnderflowException + no pool leak ──────────

    /**
     * R2-1 — Proves the BUG: when uBuffer.limit < uBuffer.capacity(), copying `capacity` bytes
     * throws BufferUnderflowException AND leaks the acquired FrameSlot from the pool.
     *
     * RED state (before fix): processFrame throws BufferUnderflowException.
     * GREEN state (after fix): no exception thrown; pool slot returned and reused across frames.
     *
     * Pool-leak assertion: frames 1, 2, and 3 all reuse the same FrameSlot object from the
     * free pool — if the slot had been leaked by frame 2, frame 3 would allocate a new slot
     * (different object reference) instead of reusing the same one.
     */
    @Test
    fun processFrame_limitedUBuffer_noBufferUnderflowThrown() {
        val capturedSlots = mutableListOf<FrameSlot>()
        val lm = MyHandLandmarker(null) { _, _, _ -> null }
        lm.processSlotFn = { slot -> capturedSlots.add(slot) }

        // Frame 1: valid 1×1 frame — seeds the free pool with one slot
        lm.processFrame(
            ByteBuffer.wrap(ByteArray(1) { 100.toByte() }),
            ByteBuffer.wrap(ByteArray(1) { 128.toByte() }),
            ByteBuffer.wrap(ByteArray(1) { 128.toByte() }),
            1, 1, 1, 1, 1, 0, 1L
        )
        lm.awaitWorkerIdleForTest(1000)
        assertEquals(1, capturedSlots.size, "frame 1 must be processed")

        // Frame 2: U/V buffers with limit=50 < capacity=100.
        // RED (before fix): uBuffer.get(slot.uArr, 0, capacity=100) -> BufferUnderflowException.
        // GREEN (after fix): uBuffer.get(slot.uArr, 0, remaining()=50) -> succeeds.
        val largeUBuf = ByteBuffer.allocate(100).also { buf ->
            repeat(50) { buf.put(128.toByte()) }
            buf.flip()  // position=0, limit=50, capacity=100
        }
        val largeVBuf = ByteBuffer.allocate(100).also { buf ->
            repeat(50) { buf.put(128.toByte()) }
            buf.flip()
        }
        // This call must NOT throw — RED if it does
        lm.processFrame(
            ByteBuffer.wrap(ByteArray(1) { 0 }),
            largeUBuf,
            largeVBuf,
            1, 1, 1, 1, 1, 0, 2L
        )
        lm.awaitWorkerIdleForTest(1000)

        // Frame 3: valid — verifies pool slot was returned after frame 2 (no leak)
        lm.processFrame(
            ByteBuffer.wrap(ByteArray(1) { 0 }),
            ByteBuffer.wrap(ByteArray(1) { 128.toByte() }),
            ByteBuffer.wrap(ByteArray(1) { 128.toByte() }),
            1, 1, 1, 1, 1, 0, 3L
        )
        lm.awaitWorkerIdleForTest(1000)
        lm.close()

        assertEquals(3, capturedSlots.size, "all three frames must reach processSlotFn")
        assertSame(capturedSlots[0], capturedSlots[1],
            "frame 2 must reuse pool slot from frame 1 (limited U buffer handled correctly)")
        assertSame(capturedSlots[0], capturedSlots[2],
            "frame 3 must reuse same pool slot — no leak introduced by limited-buffer frame 2")
    }

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
