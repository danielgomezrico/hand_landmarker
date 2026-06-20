package io.github.iot_gamer.hand_landmarker

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Unit tests for [LandmarksEncoder]. Pure JVM — no Android imports.
 * Input built via public NormalizedLandmark.create(x,y,z) (JVM-constructible in tasks-core 0.10.29).
 */
internal class LandmarksEncoderTest {

    @Test
    fun encodeZeroHands_returnsEmptyList() {
        val result = LandmarksEncoder.encode(emptyList())
        assertTrue(result.isEmpty(), "0 hands must produce empty list")
    }

    @Test
    fun encodeOneHand21Landmarks_returns63FloatArray() {
        val landmarks = (0 until 21).map { i ->
            NormalizedLandmark.create(i.toFloat(), (i + 0.1f), (i + 0.2f))
        }
        val result = LandmarksEncoder.encode(listOf(landmarks))
        assertEquals(1, result.size, "1 hand -> list size 1")
        assertEquals(63, result[0].size, "21 landmarks * 3 -> 63 floats")
        // Assert x,y,z order at indices 0,1,2 for first landmark
        assertEquals(0.0f, result[0][0], "index 0 = x of landmark 0")
        assertEquals(0.1f, result[0][1], 1e-5f, "index 1 = y of landmark 0")
        assertEquals(0.2f, result[0][2], 1e-5f, "index 2 = z of landmark 0")
        // Assert landmark 1
        assertEquals(1.0f, result[0][3], "index 3 = x of landmark 1")
        assertEquals(1.1f, result[0][4], 1e-5f, "index 4 = y of landmark 1")
        assertEquals(1.2f, result[0][5], 1e-5f, "index 5 = z of landmark 1")
    }

    @Test
    fun encodeTwoHands_returnsTwoArraysOf63() {
        val hand = (0 until 21).map { NormalizedLandmark.create(0f, 0f, 0f) }
        val result = LandmarksEncoder.encode(listOf(hand, hand))
        assertEquals(2, result.size, "2 hands -> list size 2")
        assertEquals(63, result[0].size)
        assertEquals(63, result[1].size)
    }

    @Test
    fun encodeEmptyResultLandmarks_returnsEmptyList() {
        // Simulates a result.landmarks() that is non-null but empty
        val result = LandmarksEncoder.encode(emptyList<List<NormalizedLandmark>>())
        assertTrue(result.isEmpty())
    }
}
