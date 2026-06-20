package io.github.iot_gamer.hand_landmarker

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Unit tests for [TimestampGate] — the pure monotonic-drop decision routed into
 * production `MyHandLandmarker.processFrame` (tested == shipped). Android-free.
 */
internal class MonotonicTimestampTest {

    // Thin counter wrapping only a call-count check, avoids mocking Android classes
    private var detectCallCount = 0

    private fun callDetectIfMonotonic(ts: Long, last: Long) {
        if (TimestampGate.shouldProcess(ts, last)) {
            detectCallCount++
        }
    }

    @Test
    fun equalTimestamp_shouldNotProcess() {
        assertFalse(TimestampGate.shouldProcess(100L, 100L), "equal ts must be rejected")
    }

    @Test
    fun regressedTimestamp_shouldNotProcess() {
        assertFalse(TimestampGate.shouldProcess(99L, 100L), "regressed ts must be rejected")
    }

    @Test
    fun increasingTimestamp_shouldProcess() {
        assertTrue(TimestampGate.shouldProcess(101L, 100L), "increasing ts must be accepted")
    }

    @Test
    fun equalTsThenIncreasing_detectCalledOnce() {
        detectCallCount = 0
        // equal ts -> gate rejects
        callDetectIfMonotonic(100L, 100L)
        // increasing ts -> gate accepts
        callDetectIfMonotonic(200L, 100L)
        assertTrue(detectCallCount == 1, "equal ts skipped; increasing ts fired once")
    }

    @Test
    fun twoIncreasingTs_detectCalledTwice() {
        detectCallCount = 0
        callDetectIfMonotonic(100L, -1L)
        callDetectIfMonotonic(200L, 100L)
        assertTrue(detectCallCount == 2, "two increasing ts -> detect called twice")
    }
}
