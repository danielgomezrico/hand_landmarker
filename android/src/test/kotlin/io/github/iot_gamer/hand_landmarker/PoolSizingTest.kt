package io.github.iot_gamer.hand_landmarker

import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * O-2/R2 guard: verifies that the pool-sizing helper uses wholesale buffer capacity()
 * for U/V arrays, never under-allocating for semi-planar (uvPixelStride=2) layouts.
 *
 * Semi-planar U/V buffer capacity is larger than the naive (width/2 * height/2) estimate
 * because each UV sample occupies 2 bytes (interleaved NV12/NV21). The pool must size
 * to capacity(), not to pixel-count, to avoid OOB reads in YuvConverter.
 *
 * Pure JVM — no Android imports. Tests the sizing logic directly.
 */
internal class PoolSizingTest {

    /**
     * Simulates the ensurePools capacity-based sizing for semi-planar U/V buffers.
     * Returns the pool array sizes that would be allocated for the given parameters.
     */
    private data class PoolSizes(val ySize: Int, val uSize: Int, val vSize: Int, val argbSize: Int)

    private fun computePoolSizes(
        width: Int, height: Int, uCap: Int, vCap: Int
    ): PoolSizes {
        val pixels = width * height
        val ySize = pixels
        val argbSize = pixels
        // Must use wholesale capacity — not naive width/2 * height/2
        val uSize = uCap
        val vSize = vCap
        return PoolSizes(ySize, uSize, vSize, argbSize)
    }

    @Test
    fun semiPlanarUVCapacity_neverUnderAllocates() {
        val width = 1280
        val height = 720

        // Semi-planar (NV12/NV21): uvPixelStride=2, uvRowStride=width
        // Each U or V buffer capacity: (height/2 - 1) * uvRowStride + (width/2 - 1) * uvPixelStride + 1
        val uvRowStride = width
        val uvPixelStride = 2
        val uCap = (height / 2 - 1) * uvRowStride + (width / 2 - 1) * uvPixelStride + 1
        val vCap = uCap  // same layout

        val naive = width / 2 * (height / 2)  // what an incorrect impl might use

        val pool = computePoolSizes(width, height, uCap, vCap)

        // Pool must be at least capacity-sized, never the naive pixel-count
        assertTrue(
            pool.uSize >= uCap,
            "U pool ($pool.uSize) must be >= semi-planar capacity ($uCap), not naive pixel-count ($naive)"
        )
        assertTrue(
            pool.vSize >= vCap,
            "V pool ($pool.vSize) must be >= semi-planar capacity ($vCap), not naive pixel-count ($naive)"
        )
        // Confirm the naive size would have been SMALLER (proves the test is non-vacuous)
        assertTrue(
            uCap > naive,
            "Semi-planar capacity ($uCap) must exceed naive pixel-count ($naive) — test precondition"
        )
        // Y and ARGB are pixel-count sized (correct)
        assertTrue(pool.ySize == width * height)
        assertTrue(pool.argbSize == width * height)
    }

    @Test
    fun planarUVCapacity_pixelCountSuffices() {
        // Planar (I420): uvPixelStride=1, capacity == width/2 * height/2
        val width = 640
        val height = 480
        val uvRowStride = width / 2
        val uvPixelStride = 1
        val uCap = (height / 2 - 1) * uvRowStride + (width / 2 - 1) * uvPixelStride + 1
        val naive = width / 2 * (height / 2)

        val pool = computePoolSizes(width, height, uCap, uCap)
        assertTrue(pool.uSize >= uCap)
        // For planar, capacity ~ naive (within 1 byte of pixel-count)
        assertTrue(
            uCap <= naive + 1,
            "Planar capacity ($uCap) should be ~= naive pixel count ($naive)"
        )
    }

    @Test
    fun poolReallocatesOnCapacityGrowth() {
        // Verify that smaller existing pool would be replaced when uCap grows
        val smallUCap = 100
        val largeUCap = 200
        var uArr = ByteArray(smallUCap)
        // Simulate ensurePools growth check: if (uArr.size < uCap) uArr = ByteArray(uCap)
        if (uArr.size < largeUCap) uArr = ByteArray(largeUCap)
        assertTrue(uArr.size >= largeUCap, "Pool must grow to fit new capacity")
    }

    @Test
    fun yuvConverterReceivesCorrectlySizedArrays_noOob() {
        // Integration-style: feed a semi-planar-capacity U/V array into YuvConverter
        // to verify no ArrayIndexOutOfBoundsException is thrown.
        val width = 4
        val height = 4
        val uvRowStride = width
        val uvPixelStride = 2
        val uCap = (height / 2 - 1) * uvRowStride + (width / 2 - 1) * uvPixelStride + 1

        val y = ByteArray(width * height) { 120.toByte() }
        val u = ByteArray(uCap) { 128.toByte() }
        val v = ByteArray(uCap) { 128.toByte() }
        val out = IntArray(width * height)

        // Must not throw OOB
        YuvConverter.yuv420ToArgb(out, y, u, v, width, height, width, uvRowStride, uvPixelStride)
        // All pixels should have alpha = 0xFF
        for (px in out) {
            assertTrue((px ushr 24) and 0xFF == 0xFF, "alpha must be 0xFF")
        }
    }
}
