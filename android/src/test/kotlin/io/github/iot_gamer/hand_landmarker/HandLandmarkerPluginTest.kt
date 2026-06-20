package io.github.iot_gamer.hand_landmarker

import io.flutter.plugin.common.EventChannel
import kotlin.test.Test
import org.mockito.Mockito.mock
import org.mockito.Mockito.never
import org.mockito.Mockito.verify

/**
 * Unit tests for [emitError] — the pure-core, Android-import-free top-level function.
 * Uses mockito-core 5.0.0 (interfaces only, no inline mock-maker).
 * Does NOT instantiate HandLandmarkerPlugin (companion Handler(Looper) crashes in JVM stub).
 */
internal class HandLandmarkerPluginTest {

    @Test
    fun emitError_callsSinkErrorWithCorrectArgs() {
        val sink: EventChannel.EventSink = mock(EventChannel.EventSink::class.java)
        emitError(sink, "MEDIAPIPE_ERROR", "something went wrong")
        verify(sink).error("MEDIAPIPE_ERROR", "something went wrong", null)
    }

    @Test
    fun emitError_nullSink_doesNotThrow() {
        // Must not throw — null sink is a no-op
        emitError(null, "MEDIAPIPE_ERROR", "something went wrong")
    }

    @Test
    fun emitError_sinkErrorCalledExactlyOnce() {
        val sink: EventChannel.EventSink = mock(EventChannel.EventSink::class.java)
        emitError(sink, "MEDIAPIPE_ERROR", "msg")
        verify(sink).error("MEDIAPIPE_ERROR", "msg", null)
        // Verify success/endOfStream are NOT called
        verify(sink, never()).success(org.mockito.ArgumentMatchers.any())
        verify(sink, never()).endOfStream()
    }
}
