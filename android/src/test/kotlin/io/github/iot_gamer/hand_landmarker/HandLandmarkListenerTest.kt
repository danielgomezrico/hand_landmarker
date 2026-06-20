package io.github.iot_gamer.hand_landmarker

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * K4 — tests that HandLandmarkListener receives the correct List<FloatArray> payload
 * assembled by LandmarksEncoder. Uses a hand-rolled fake listener (RC-7: no inline mocks).
 */
internal class HandLandmarkListenerTest {

    private class FakeListener : HandLandmarkListener {
        var receivedHands: List<FloatArray>? = null
        var receivedErrorCode: String? = null
        var receivedErrorMsg: String? = null

        override fun onLandmarksDetected(hands: List<FloatArray>) {
            receivedHands = hands
        }

        override fun onError(code: String, message: String) {
            receivedErrorCode = code
            receivedErrorMsg = message
        }
    }

    @Test
    fun fakeListenerReceivesEncodedPayload_oneHand() {
        val fake = FakeListener()
        val landmarks = (0 until 21).map { i ->
            NormalizedLandmark.create(i.toFloat(), (i + 0.1f), (i + 0.2f))
        }
        // Simulate what MyHandLandmarker's result listener does
        val encoded = LandmarksEncoder.encode(listOf(landmarks))
        fake.onLandmarksDetected(encoded)

        assertNotNull(fake.receivedHands)
        assertEquals(1, fake.receivedHands!!.size)
        assertEquals(63, fake.receivedHands!![0].size)
        assertEquals(0.0f, fake.receivedHands!![0][0])
        assertEquals(0.1f, fake.receivedHands!![0][1], 1e-5f)
        assertEquals(0.2f, fake.receivedHands!![0][2], 1e-5f)
    }

    @Test
    fun fakeListenerReceivesEmptyList_whenNoHands() {
        val fake = FakeListener()
        fake.onLandmarksDetected(emptyList())
        assertNotNull(fake.receivedHands)
        assertTrue(fake.receivedHands!!.isEmpty())
    }

    @Test
    fun fakeListenerReceivesError() {
        val fake = FakeListener()
        fake.onError("MEDIAPIPE_ERROR", "test error")
        assertEquals("MEDIAPIPE_ERROR", fake.receivedErrorCode)
        assertEquals("test error", fake.receivedErrorMsg)
    }

    @Test
    fun fakeListenerReceivesTwoHands() {
        val fake = FakeListener()
        val hand = (0 until 21).map { NormalizedLandmark.create(0f, 0f, 0f) }
        val encoded = LandmarksEncoder.encode(listOf(hand, hand))
        fake.onLandmarksDetected(encoded)
        assertEquals(2, fake.receivedHands!!.size)
        assertEquals(63, fake.receivedHands!![0].size)
        assertEquals(63, fake.receivedHands!![1].size)
    }
}
