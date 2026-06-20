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
}
