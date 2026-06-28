package io.github.iot_gamer.hand_landmarker

import java.util.Arrays
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regression tests for the fixed-point integer YUV_420_888 -> ARGB conversion
 * ([YuvConverter.yuv420ToArgb]). Pure JVM — no Android, Bitmap, or MediaPipe.
 *
 * Run from `example/android/`: `./gradlew :hand_landmarker:testDebugUnitTest`.
 */
internal class YuvConverterTest {

    private fun b(value: Int): Byte = value.toByte()

    private fun r(argb: Int) = (argb ushr 16) and 0xFF
    private fun g(argb: Int) = (argb ushr 8) and 0xFF
    private fun bch(argb: Int) = argb and 0xFF
    private fun a(argb: Int) = (argb ushr 24) and 0xFF

    /** Builds a 2x2 frame of uniform (Y, U, V) and returns the top-left ARGB pixel. */
    private fun uniformPixel(yVal: Int, uVal: Int, vVal: Int): Int {
        val width = 2
        val height = 2
        val y = ByteArray(width * height) { b(yVal) } // tight, stride == width
        val u = ByteArray(1) { b(uVal) }
        val v = ByteArray(1) { b(vVal) }
        val out = IntArray(width * height)
        YuvConverter.yuv420ToArgb(
            out, y, u, v, width, height,
            yStride = width, uvRowStride = 1, uvPixelStride = 1
        )
        return out[0]
    }

    // Round-half-up that is correct for negative numerators (matches `(x + 512) shr 10`).
    private fun roundShift(x: Int): Int = Math.floorDiv(x + 512, 1024)

    @Test
    fun alphaIsAlwaysOpaque() {
        assertEquals(0xFF, a(uniformPixel(0, 128, 128)))
        assertEquals(0xFF, a(uniformPixel(255, 0, 255)))
    }

    @Test
    fun blackWhiteGrayAreExact() {
        val black = uniformPixel(0, 128, 128)
        assertEquals(0, r(black)); assertEquals(0, g(black)); assertEquals(0, bch(black))
        val white = uniformPixel(255, 128, 128)
        assertEquals(255, r(white)); assertEquals(255, g(white)); assertEquals(255, bch(white))
        val gray = uniformPixel(128, 128, 128)
        assertEquals(128, r(gray)); assertEquals(128, g(gray)); assertEquals(128, bch(gray))
    }

    @Test
    fun saturatedChromaClampsInRange() {
        for (yVal in intArrayOf(0, 128, 255)) {
            for (uVal in intArrayOf(0, 255)) {
                for (vVal in intArrayOf(0, 255)) {
                    val p = uniformPixel(yVal, uVal, vVal)
                    assertTrue(r(p) in 0..255 && g(p) in 0..255 && bch(p) in 0..255)
                }
            }
        }
    }

    /**
     * A1+A2 — gap-free over the full chroma plane (every (U,V) in 256x256), at the
     * clamp boundaries and mid-range for Y, asserting the conversion equals the
     * correctly ROUNDED (round-to-nearest) fixed-point value — not a truncated one.
     * Buffers are allocated once and reused; only the Y plane is refilled per Y.
     */
    @Test
    fun roundToNearestExactAcrossFullChromaPlane() {
        val width = 512
        val height = 512
        // chroma block (col,row) encodes U=col, V=row over the full 0..255 x 0..255 cube.
        val y = ByteArray(width * height)
        val u = ByteArray(256 * 256)
        val v = ByteArray(256 * 256)
        for (row in 0 until 256) {
            for (col in 0 until 256) {
                u[row * 256 + col] = b(col)
                v[row * 256 + col] = b(row)
            }
        }
        val out = IntArray(width * height)

        var maxErr = 0
        for (yVal in intArrayOf(0, 1, 64, 128, 191, 254, 255)) {
            Arrays.fill(y, b(yVal))
            YuvConverter.yuv420ToArgb(out, y, u, v, width, height, width, 256, 1)
            for (row in 0 until 256) {
                for (col in 0 until 256) {
                    val uu = col - 128
                    val vv = row - 128
                    val refR = (yVal + roundShift(1436 * vv)).coerceIn(0, 255)
                    val refG = (yVal - roundShift(352 * uu + 731 * vv)).coerceIn(0, 255)
                    val refB = (yVal + roundShift(1815 * uu)).coerceIn(0, 255)
                    val px = out[(row * 2) * width + (col * 2)]
                    val er = abs(r(px) - refR)
                    val eg = abs(g(px) - refG)
                    val eb = abs(bch(px) - refB)
                    maxErr = maxOf(maxErr, er, eg, eb)
                    assertEquals(refR, r(px), "R mismatch at Y=$yVal U=$col V=$row")
                    assertEquals(refG, g(px), "G mismatch at Y=$yVal U=$col V=$row")
                    assertEquals(refB, bch(px), "B mismatch at Y=$yVal U=$col V=$row")
                }
            }
        }
        assertEquals(0, maxErr, "conversion must equal the round-to-nearest fixed-point value")
    }

    /**
     * Accuracy bound: the rounded fixed-point output stays within 1 LSB of the
     * exact real-valued BT.601 (JFIF full-range) conversion across the chroma plane.
     */
    @Test
    fun within1LsbOfExactRealConversion() {
        var maxErr = 0
        var uVal = 0
        while (uVal <= 255) {
            var vVal = 0
            while (vVal <= 255) {
                // Y mid-range so the comparison isn't dominated by clamping.
                val px = uniformPixel(128, uVal, vVal)
                val uu = uVal - 128
                val vv = vVal - 128
                val exactR = (128 + 1.402 * vv).roundToInt().coerceIn(0, 255)
                val exactG = (128 - 0.344136 * uu - 0.714136 * vv).roundToInt().coerceIn(0, 255)
                val exactB = (128 + 1.772 * uu).roundToInt().coerceIn(0, 255)
                maxErr = maxOf(maxErr, abs(r(px) - exactR), abs(g(px) - exactG), abs(bch(px) - exactB))
                vVal += 1
            }
            uVal += 1
        }
        assertTrue(maxErr <= 1, "max channel error vs exact real conversion was $maxErr (want <=1)")
    }

    @Test
    fun planarAndSemiPlanarLayoutsProduceSameColors() {
        val width = 4
        val height = 4
        val blockU = intArrayOf(40, 200, 128, 90)
        val blockV = intArrayOf(220, 60, 128, 150)
        val y = ByteArray(width * height) { b(150) }

        val uP = ByteArray(4)
        val vP = ByteArray(4)
        for (k in 0 until 4) { uP[k] = b(blockU[k]); vP[k] = b(blockV[k]) }
        val outP = IntArray(width * height)
        YuvConverter.yuv420ToArgb(outP, y, uP, vP, width, height, yStride = width, uvRowStride = 2, uvPixelStride = 1)

        val uS = ByteArray(8)
        val vS = ByteArray(8)
        for (row in 0 until 2) {
            for (col in 0 until 2) {
                val idx = row * 4 + col * 2
                uS[idx] = b(blockU[row * 2 + col])
                vS[idx] = b(blockV[row * 2 + col])
            }
        }
        val outS = IntArray(width * height)
        YuvConverter.yuv420ToArgb(outS, y, uS, vS, width, height, yStride = width, uvRowStride = 4, uvPixelStride = 2)

        assertTrue(outP.contentEquals(outS), "planar and semi-planar layouts must yield identical pixels")
        assertTrue(outP.toSet().size >= 3, "distinct chroma blocks should produce distinct colors")
    }

    /** A3 — non-square semi-planar frame with planes sized exactly to capacity: no OOB. */
    @Test
    fun semiPlanarTightPlanesNonSquareNoOob() {
        val width = 6
        val height = 4
        val y = ByteArray(width * height) { b(120) }
        // chroma grid 3x2; semi-planar pixelStride=2, rowStride=width(6); last sample at
        // (1)*6 + (2)*2 = 10, +1 byte => capacity 12.
        val uvRowStride = width
        val uvPixelStride = 2
        val cap = (height / 2 - 1) * uvRowStride + (width / 2 - 1) * uvPixelStride + 1
        val u = ByteArray(cap) { b(110) }
        val v = ByteArray(cap) { b(140) }
        val out = IntArray(width * height)
        // Must not throw and must fill every pixel (alpha set on all).
        YuvConverter.yuv420ToArgb(out, y, u, v, width, height, width, uvRowStride, uvPixelStride)
        for (px in out) assertEquals(0xFF, a(px))
    }

    /** A4 — a zero-size frame is a no-op and never throws. */
    @Test
    fun zeroSizeIsNoOp() {
        val out = IntArray(0)
        YuvConverter.yuv420ToArgb(out, ByteArray(0), ByteArray(0), ByteArray(0), 0, 0, 0, 0, 1)
        // width=0 with positive height: still no writes, no throw.
        val out2 = IntArray(4)
        YuvConverter.yuv420ToArgb(out2, ByteArray(0), ByteArray(0), ByteArray(0), 0, 2, 0, 1, 1)
        assertTrue(out2.all { it == 0 }, "no pixels should be written for width=0")
    }

    /** A5 — an oversized out buffer (reused across frames) has only its first w*h ints written. */
    @Test
    fun oversizedOutOnlyFirstNWritten() {
        val width = 2
        val height = 2
        val sentinel = 0x0BADF00D
        val out = IntArray(width * height + 3) { sentinel }
        val y = ByteArray(width * height) { b(200) }
        val u = ByteArray(1) { b(128) }
        val v = ByteArray(1) { b(128) }
        YuvConverter.yuv420ToArgb(out, y, u, v, width, height, width, 1, 1)
        for (i in 0 until width * height) assertEquals(0xFF, a(out[i]))
        for (i in width * height until out.size) assertEquals(sentinel, out[i], "tail beyond w*h must be untouched")
    }

    // ── GUARD: padded Y stride ────────────────────────────────────────────────────────────────

    /**
     * G1 — yStride padding: with yStride=8 and width=4 the Y plane has 4 padding bytes per
     * row. Every pixel must be read from y[row*yStride + col], NOT y[row*width + col].
     * Padding bytes are set to a poison value (99) to make any incorrect indexing visible.
     */
    @Test
    fun yStridePadding_correctLumaNoOob() {
        val width = 4; val height = 2; val yStride = 8
        // Poison the whole array (99), then stamp correct lumas at padded positions.
        val y = ByteArray(yStride * height) { 99.toByte() }
        val lumaRow0 = byteArrayOf(10, 20, 30, 40)
        val lumaRow1 = byteArrayOf(50, 60, 70, 80)
        for (col in 0 until width) {
            y[0 * yStride + col] = lumaRow0[col]
            y[1 * yStride + col] = lumaRow1[col]
        }
        // Neutral UV (U=V=128) → R=G=B=Y for each pixel.
        // width=4: chroma cols 0 and 1 accessed (i shr 1 ∈ {0,1}); uvRowStride=2 for 2 chroma cols.
        val u = ByteArray(2) { 128.toByte() }
        val v = ByteArray(2) { 128.toByte() }
        val out = IntArray(width * height)
        YuvConverter.yuv420ToArgb(out, y, u, v, width, height, yStride = yStride, uvRowStride = 2, uvPixelStride = 1)
        for (col in 0 until width) {
            val px0 = out[0 * width + col]
            assertEquals(0xFF, a(px0), "alpha row=0 col=$col")
            assertEquals(lumaRow0[col].toInt() and 0xFF, r(px0),
                "R row=0 col=$col must use y[0*$yStride+$col] not poison byte")
            val px1 = out[1 * width + col]
            assertEquals(0xFF, a(px1), "alpha row=1 col=$col")
            assertEquals(lumaRow1[col].toInt() and 0xFF, r(px1),
                "R row=1 col=$col must use y[1*$yStride+$col] not poison byte")
        }
    }

    // ── GUARD: odd width ─────────────────────────────────────────────────────────────────────

    /**
     * G2 — odd width=3: last pixel column (col=2) maps to uvCol=(2 shr 1)*uvPixelStride=1,
     * while cols 0 and 1 both map to uvCol=0. UV arrays are sized to the minimum tight
     * capacity needed for ceil(width/2)=2 chroma columns — no ArrayIndexOutOfBoundsException.
     */
    @Test
    fun oddWidth3_lastColSharesChromaWithCol1_noOob() {
        val width = 3; val height = 2
        // uvRowStride=1, uvPixelStride=1: tight planar layout.
        // Max UV index = (2 shr 1)*1 = 1 → minimum capacity = 2.
        val uvRowStride = 1; val uvPixelStride = 1
        val y = ByteArray(width * height) { 100.toByte() }
        // Chroma col 0 (neutral) → used by pixel cols 0 and 1.
        // Chroma col 1 (shifted) → used by pixel col 2 only.
        val u = byteArrayOf(128.toByte(), 50.toByte())
        val v = byteArrayOf(128.toByte(), 128.toByte())
        val out = IntArray(width * height)
        YuvConverter.yuv420ToArgb(out, y, u, v, width, height,
            yStride = width, uvRowStride = uvRowStride, uvPixelStride = uvPixelStride)
        // No OOB: test would throw ArrayIndexOutOfBoundsException above if UV capacity were wrong.
        for (px in out) assertEquals(0xFF, a(px), "all 6 pixels must have alpha=0xFF")
        // Cols 0 and 1 share chroma col 0 (both read u[0]); same Y → identical ARGB.
        assertEquals(out[0], out[1], "col=0 and col=1 share chroma col 0 → same ARGB (row 0)")
        assertEquals(out[width], out[width + 1], "col=0 and col=1 share chroma col 0 → same ARGB (row 1)")
        // Col=2 uses chroma col 1 (u[1]=50 ≠ u[0]=128) → different ARGB.
        assertTrue(out[2] != out[0], "col=2 uses a distinct chroma sample → different ARGB from col=0")
    }

    // ── GUARD: odd height ────────────────────────────────────────────────────────────────────

    /**
     * G3 — odd height=3: last pixel row (row=2) maps to uvRow=(2 shr 1)*uvRowStride=uvRowStride,
     * which is the SECOND chroma row. UV arrays must be sized for ceil(height/2)=2 chroma rows.
     * Stale single-row sizing would cause OOB at row=2.
     */
    @Test
    fun oddHeight3_lastRowReadsSecondUvRow_noOob() {
        val width = 4; val height = 3
        // Tight planar: uvRowStride = width/2 = 2, uvPixelStride = 1.
        // Max UV index = (2 shr 1)*2 + (3 shr 1)*1 = 2 + 1 = 3 → capacity 4.
        val uvRowStride = 2; val uvPixelStride = 1
        val y = ByteArray(width * height) { 100.toByte() }
        // Chroma row 0 (u[0..1]) = neutral (128) → rows 0 and 1 neutral.
        // Chroma row 1 (u[2..3]) = shifted (50, 128) → row 2 uses different chroma.
        val u = byteArrayOf(b(128), b(128), b(50), b(128))
        val v = byteArrayOf(b(128), b(128), b(128), b(128))
        val out = IntArray(width * height)
        YuvConverter.yuv420ToArgb(out, y, u, v, width, height,
            yStride = width, uvRowStride = uvRowStride, uvPixelStride = uvPixelStride)
        // No OOB: test would throw above if UV capacity were only 1 chroma row.
        for (px in out) assertEquals(0xFF, a(px), "all 12 pixels must have alpha=0xFF")
        // Rows 0 and 1 share chroma row 0 (neutral) → same ARGB as each other.
        assertEquals(out[0], out[width], "row=0 and row=1 same neutral chroma → identical pixel (col=0)")
        // Row 2 uses chroma row 1 (u[2]=50, uu=-78) → Blue channel shifts → distinct from row 0.
        assertTrue(out[2 * width] != out[0],
            "row=2 uses chroma row 1 (u=50) → different ARGB from rows 0/1")
    }

    // ── GUARD: degenerate frames ─────────────────────────────────────────────────────────────

    /**
     * G7 — 1×4 and 4×1 frames: extreme camera crops that are technically valid
     * YUV_420_888 configurations. Every pixel must map to a valid UV sample and
     * no ArrayIndexOutOfBoundsException must occur.
     */
    @Test
    fun degenerateFrames_width1height4_and_width4height1_noOob() {
        // width=1, height=4: every pixel uses uvCol=0; uvRow alternates 0 and uvRowStride.
        run {
            val width = 1; val height = 4
            val uvRowStride = 1; val uvPixelStride = 1
            // Max UV index = ((3) shr 1)*1 + 0 = 1 → capacity 2.
            val y = ByteArray(width * height) { 120.toByte() }
            val u = ByteArray(2) { 128.toByte() }
            val v = ByteArray(2) { 128.toByte() }
            val out = IntArray(width * height)
            YuvConverter.yuv420ToArgb(out, y, u, v, width, height,
                yStride = width, uvRowStride = uvRowStride, uvPixelStride = uvPixelStride)
            for (px in out) assertEquals(0xFF, a(px), "width=1,height=4: all pixels alpha=0xFF")
        }
        // width=4, height=1: every pixel uses uvRow=0; uvCol ranges over chroma cols 0..1.
        run {
            val width = 4; val height = 1
            val uvRowStride = 1; val uvPixelStride = 1
            // Max UV index = 0 + ((3) shr 1)*1 = 1 → capacity 2.
            val y = ByteArray(width * height) { 120.toByte() }
            val u = ByteArray(2) { 128.toByte() }
            val v = ByteArray(2) { 128.toByte() }
            val out = IntArray(width * height)
            YuvConverter.yuv420ToArgb(out, y, u, v, width, height,
                yStride = width, uvRowStride = uvRowStride, uvPixelStride = uvPixelStride)
            for (px in out) assertEquals(0xFF, a(px), "width=4,height=1: all pixels alpha=0xFF")
        }
    }

    // ── GUARD: degenerate 1×1 ────────────────────────────────────────────────────────────────

    /**
     * G8 — 1×1 frame: minimum valid frame. Exactly one pixel produced; its ARGB must
     * match the BT.601 fixed-point formula within the same round-to-nearest contract
     * as the full-sweep test.
     */
    @Test
    fun degenerate1x1_singlePixelCorrectArgb() {
        val yVal = 128; val uVal = 100; val vVal = 200
        val out = IntArray(1)
        YuvConverter.yuv420ToArgb(
            out,
            y = byteArrayOf(yVal.toByte()),
            u = byteArrayOf(uVal.toByte()),
            v = byteArrayOf(vVal.toByte()),
            width = 1, height = 1,
            yStride = 1, uvRowStride = 1, uvPixelStride = 1
        )
        val uu = uVal - 128  // -28
        val vv = vVal - 128  //  72
        val expectedR = (yVal + roundShift(1436 * vv)).coerceIn(0, 255)           // 229
        val expectedG = (yVal - roundShift(352 * uu + 731 * vv)).coerceIn(0, 255) //  86
        val expectedB = (yVal + roundShift(1815 * uu)).coerceIn(0, 255)           //  78
        assertEquals(0xFF, a(out[0]), "1x1 pixel must have alpha=0xFF")
        assertEquals(expectedR, r(out[0]), "R must match BT.601 for Y=$yVal U=$uVal V=$vVal")
        assertEquals(expectedG, g(out[0]), "G must match BT.601 for Y=$yVal U=$uVal V=$vVal")
        assertEquals(expectedB, bch(out[0]), "B must match BT.601 for Y=$yVal U=$uVal V=$vVal")
    }
}
