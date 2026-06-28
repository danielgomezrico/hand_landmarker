package io.github.iot_gamer.hand_landmarker

import kotlin.test.Test
import kotlin.test.assertNotSame
import kotlin.test.assertSame

/**
 * Unit tests for [ensureArgbCapacity] — the ARGB buffer sizing seam extracted from
 * [MyHandLandmarker.processSlot].
 *
 * These tests target the high-water-mark fix (RF-1) for the per-frame allocation
 * churn caused by the original dim-equality check.
 *
 * Pure JVM — no Android, no Bitmap, no MediaPipe.
 *
 * Run from `example/android/`: `./gradlew :hand_landmarker:testDebugUnitTest`.
 */
internal class ArgbBufferTest {

    // ── RF-1 RED test (behavioral — must FAIL against dim-based policy) ──────────────────────

    /**
     * RF-1 RED — rotation W×H ↔ H×W with same pixel count must NOT reallocate.
     *
     * [ensureArgbCapacity] currently reallocates whenever (width, height) differs from
     * (prevWidth, prevHeight), even when w*h is unchanged (e.g., 2×8 → 8×2). This causes
     * per-frame IntArray allocation churn when the camera rotates between landscape and
     * portrait at the same resolution — visible as GC pressure at 60 fps.
     *
     * HIGH-WATER-MARK FIX (applied in the next commit): only reallocate when
     * `current.size < width * height`. A larger or equal buffer is safe to reuse because
     * `Bitmap.createBitmap(argbArray, width, height, …)` reads only the first w*h ints.
     *
     * **Expected RED against dim-based policy:** the dim-based implementation returns a NEW
     * IntArray for 2×8 → 8×2 (dims differ), so assertSame fails — proving the test would
     * catch the regression before the fix is applied.
     */
    @Test
    fun argbBuffer_rotationSamePixelCount_doesNotReallocate() {
        // Simulate a prior 2×8 frame (16 pixels) → allocate a 16-element buffer.
        val initial = ensureArgbCapacity(IntArray(0), width = 2, height = 8, prevWidth = 0, prevHeight = 0)
        // Rotating to 8×2 — same pixel count (16) — must NOT produce a new array.
        val rotated = ensureArgbCapacity(initial, width = 8, height = 2, prevWidth = 2, prevHeight = 8)
        assertSame(initial, rotated,
            "rotating 2x8 → 8x2 (same pixel count=16) must reuse the existing buffer; " +
            "a new IntArray(16) per frame is per-frame allocation churn at 60fps")
    }

    /**
     * RF-1 RED variant — shrink 4×4 → 2×2: current buffer (size=16) is larger than needed
     * (4 pixels). Must NOT reallocate — the tail is ignored by Bitmap.createBitmap.
     */
    @Test
    fun argbBuffer_shrink_doesNotReallocate() {
        val initial = ensureArgbCapacity(IntArray(0), width = 4, height = 4, prevWidth = 0, prevHeight = 0)
        val shrunk = ensureArgbCapacity(initial, width = 2, height = 2, prevWidth = 4, prevHeight = 4)
        assertSame(initial, shrunk,
            "shrink from 4x4 to 2x2 must reuse the existing 16-element buffer; " +
            "only first 4 ints are read by Bitmap.createBitmap for a 2x2 frame")
    }

    // ── Companion GREEN test: grow must still allocate ───────────────────────────────────────

    /**
     * Ensures the high-water-mark fix does NOT suppress legitimate grows.
     * Growing from 4×4 (16) to 8×8 (64) must allocate a new buffer.
     * This test is GREEN against BOTH the dim-based policy and the high-water-mark fix.
     */
    @Test
    fun argbBuffer_grow_reallocates() {
        val initial = ensureArgbCapacity(IntArray(0), width = 4, height = 4, prevWidth = 0, prevHeight = 0)
        val grown = ensureArgbCapacity(initial, width = 8, height = 8, prevWidth = 4, prevHeight = 4)
        assertNotSame(initial, grown,
            "grow from 4x4 to 8x8 must allocate a new buffer (64 > 16)")
    }
}
