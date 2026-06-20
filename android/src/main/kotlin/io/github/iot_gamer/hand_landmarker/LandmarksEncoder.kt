package io.github.iot_gamer.hand_landmarker

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

/**
 * Converts MediaPipe landmark results to the wire payload: List<FloatArray>
 * (one FloatArray(handSize*3) per hand, x,y,z interleaved).
 *
 * Pure Android-free object — JVM unit-testable without android.jar stubs.
 */
internal object LandmarksEncoder {

    fun encode(hands: List<List<NormalizedLandmark>>): List<FloatArray> {
        if (hands.isEmpty()) return emptyList()
        return hands.map { landmarks ->
            val arr = FloatArray(landmarks.size * 3)
            landmarks.forEachIndexed { i, lm ->
                arr[i * 3] = lm.x()
                arr[i * 3 + 1] = lm.y()
                arr[i * 3 + 2] = lm.z()
            }
            arr
        }
    }
}
