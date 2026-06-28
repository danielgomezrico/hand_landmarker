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

    // ── R2-4: extreme boundary values (zero and Long.MIN_VALUE) ──────────────────────────

    /**
     * R2-4 — GUARD: verifies [TimestampGate.shouldProcess] at extreme Long boundaries.
     *
     * - incoming=0, last=-1 (initial sentinel): 0 > -1 → accepted (first valid timestamp of 0).
     * - incoming=Long.MIN_VALUE, last=-1: MIN_VALUE < -1 → rejected (cannot be a valid camera ts).
     * - incoming=0, last=Long.MIN_VALUE: 0 > MIN_VALUE → accepted.
     *
     * Mutation proof: change `incoming > last` to `incoming < last` — the first assertTrue
     * asserts shouldProcess(0, -1)=true but gets false → RED.
     */
    @Test
    fun timestampGate_extremeBoundaries_zeroAndMinValue() {
        // incoming=0 with sentinel last=-1: first valid timestamp zero must be accepted
        assertTrue(
            TimestampGate.shouldProcess(0L, -1L),
            "timestamp 0 with last=-1 (sentinel) must be accepted (0 > -1)"
        )
        // incoming=Long.MIN_VALUE with sentinel last=-1: MIN_VALUE < -1, must be rejected
        assertFalse(
            TimestampGate.shouldProcess(Long.MIN_VALUE, -1L),
            "timestamp MIN_VALUE with last=-1 must be rejected (MIN_VALUE < -1)"
        )
        // incoming=0, last=Long.MIN_VALUE: 0 > MIN_VALUE, must be accepted
        assertTrue(
            TimestampGate.shouldProcess(0L, Long.MIN_VALUE),
            "timestamp 0 with last=MIN_VALUE must be accepted (0 > MIN_VALUE)"
        )
    }
}
