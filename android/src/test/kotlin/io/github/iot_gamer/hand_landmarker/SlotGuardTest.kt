package io.github.iot_gamer.hand_landmarker

import java.nio.ByteBuffer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Guard tests for the slot-pool array-sizing path in [MyHandLandmarker.processFrame].
 *
 * Uses the [MyHandLandmarker.processSlotFn] injection seam to intercept the slot
 * after its arrays have been resized but before any Android/MediaPipe call (which
 * would require Bitmap/Robolectric). Pure JVM — no Android classes imported.
 *
 * Run from `example/android/`: `./gradlew :hand_landmarker:testDebugUnitTest`.
 */
internal class SlotGuardTest {

    /** Create a [ByteBuffer] filled with [value], capacity=[size]. */
    private fun buf(size: Int, value: Byte = 120.toByte()): ByteBuffer =
        ByteBuffer.wrap(ByteArray(size) { value })

    /** Call processFrame with tight Y stride and planar UV, then drain synchronously. */
    private fun sendFrame(
        lm: MyHandLandmarker,
        width: Int, height: Int,
        yValue: Byte = 100.toByte(),
        uValue: Byte = 128.toByte(),
        vValue: Byte = 128.toByte(),
        timestampMs: Long
    ) {
        // Tight Y (stride == width); planar UV: ceil(w/2)*ceil(h/2) samples.
        val uvCap = ((width + 1) / 2) * ((height + 1) / 2)
        val uvRowStride = (width + 1) / 2
        lm.processFrame(
            yBuffer = buf(width * height, yValue),
            uBuffer = buf(uvCap, uValue),
            vBuffer = buf(uvCap, vValue),
            width = width, height = height,
            yRowStride = width,
            uvRowStride = uvRowStride,
            uvPixelStride = 1,
            rotation = 0,
            timestampMs = timestampMs
        )
        lm.awaitWorkerIdleForTest(1000)
    }

    // ── G4: slot shrink 4×4 → 2×2 ──────────────────────────────────────────────────────────

    /**
     * G4 — slot shrink: a pool slot whose yArr.size=16 (from a prior 4×4 frame) must NOT
     * be reallocated when a 2×2 frame arrives (pixels=4 ≤ 16). The stale tail yArr[4..15]
     * is never written by the 2×2 conversion — only indices 0..3 hold fresh luma.
     */
    @Test
    fun slotShrink_4x4to2x2_yArrNotResized_onlyNewDimsWritten() {
        val slots = mutableListOf<FrameSlot>()
        val lm = MyHandLandmarker(null) { _, _, _ -> null }
        lm.processSlotFn = { slot -> slots += slot }

        // First frame: 4×4 — yArr allocated to at least 16.
        sendFrame(lm, width = 4, height = 4, timestampMs = 1L)
        assertEquals(1, slots.size, "processSlotFn must be called after first frame")
        assertTrue(slots[0].yArr.size >= 16, "4x4 frame must produce yArr.size >= 16")

        // Second frame: 2×2 — same slot from free pool; yArr must NOT shrink.
        sendFrame(lm, width = 2, height = 2, timestampMs = 2L)
        lm.close()

        assertEquals(2, slots.size, "processSlotFn must be called after second frame")
        // Same slot object reused from free pool.
        assertSame(slots[0], slots[1], "same FrameSlot must be recycled from the free pool")
        // yArr must remain at 16 (16 >= 4, no reallocation needed).
        assertEquals(16, slots[1].yArr.size,
            "yArr must retain size=16 after 2x2 frame — high-water-mark, never shrinks")
    }

    // ── G5: slot grow 4×4 → 8×8 ────────────────────────────────────────────────────────────

    /**
     * G5 — slot grow: a pool slot with yArr.size=16 IS reallocated to 64 when an 8×8
     * frame arrives (pixels=64 > 16). All 64 bytes in the new yArr hold fresh luma.
     */
    @Test
    fun slotGrow_4x4to8x8_yArrResizedTo64() {
        val slots = mutableListOf<FrameSlot>()
        val lm = MyHandLandmarker(null) { _, _, _ -> null }
        lm.processSlotFn = { slot -> slots += slot }

        // First frame: 4×4 — yArr allocated to 16.
        sendFrame(lm, width = 4, height = 4, timestampMs = 1L)
        assertEquals(1, slots.size, "processSlotFn must be called after first frame")
        val sizeAfterFirst = slots[0].yArr.size
        assertTrue(sizeAfterFirst >= 16, "4x4 frame must produce yArr.size >= 16")

        // Second frame: 8×8 — slot.yArr.size=16 < 64 → must reallocate.
        sendFrame(lm, width = 8, height = 8, timestampMs = 2L)
        lm.close()

        assertEquals(2, slots.size, "processSlotFn must be called after second frame")
        // Same slot object reused from free pool.
        assertSame(slots[0], slots[1], "same FrameSlot must be recycled from the free pool")
        // yArr must have grown to fit 64 pixels.
        assertTrue(slots[1].yArr.size >= 64,
            "yArr must grow to >= 64 for 8x8 frame (was $sizeAfterFirst)")
    }

    // ── G6: same-dimension slot reuse ───────────────────────────────────────────────────────

    /**
     * G6 — same-dim reuse: after a 2×2 frame drains and returns the slot to the free pool,
     * the next 2×2 frame acquires the same slot without reallocating its arrays. The zero-
     * alloc-per-frame contract at steady state must hold.
     */
    @Test
    fun sameDim2x2SlotReuse_noArrayReallocation() {
        val slots = mutableListOf<FrameSlot>()
        val lm = MyHandLandmarker(null) { _, _, _ -> null }
        lm.processSlotFn = { slot -> slots += slot }

        // First frame: 2×2.
        sendFrame(lm, width = 2, height = 2, timestampMs = 1L)
        assertEquals(1, slots.size, "processSlotFn must be called after first frame")
        val yArrAfterFirst = slots[0].yArr

        // Second frame: same dims — slot from free pool, no reallocation.
        sendFrame(lm, width = 2, height = 2, timestampMs = 2L)
        lm.close()

        assertEquals(2, slots.size, "processSlotFn must be called after second frame")
        // Same slot object AND same yArr instance (no reallocation since 4 >= 4).
        assertSame(slots[0], slots[1], "same FrameSlot must be recycled from the free pool")
        assertSame(yArrAfterFirst, slots[1].yArr,
            "yArr must not be reallocated when frame dims are unchanged")
    }
}
