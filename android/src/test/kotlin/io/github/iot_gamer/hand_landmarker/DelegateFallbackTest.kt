package io.github.iot_gamer.hand_landmarker

import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * Device-free JVM tests for O1 delegate fallback logic in [MyHandLandmarker.initialize].
 *
 * The [buildHandLandmarker] seam is injected so no Android [Context], no MediaPipe model,
 * and no GPU driver are required. The builder returns null (HandLandmarker? seam) so the
 * test never instantiates the final MediaPipe class. [context] is null-cast and never
 * dereferenced by the injected builder — same isolation pattern as [PoolSizingTest].
 */
internal class DelegateFallbackTest {

    private fun landmarker(
        builder: (numHands: Int, confidence: Float, delegate: Delegate) -> HandLandmarker?
    ): MyHandLandmarker {
        // context=null: the injected builder never uses it (no MediaPipe call made).
        return MyHandLandmarker(null, builder)
    }

    @Test
    fun gpuSuccess_activeDelegateIsGpu() {
        val lm = landmarker { _, _, delegate ->
            assertEquals(Delegate.GPU, delegate)
            null
        }
        lm.initialize(1, 0.5f, useGpu = true)
        assertEquals(Delegate.GPU, lm.activeDelegate)
    }

    @Test
    fun gpuFailure_fallsBackToCpu_activeDelegateIsCpu() {
        var callCount = 0
        val lm = landmarker { _, _, delegate ->
            callCount++
            if (delegate == Delegate.GPU) throw RuntimeException("OpenCL unavailable")
            null
        }
        lm.initialize(1, 0.5f, useGpu = true)
        assertEquals(Delegate.CPU, lm.activeDelegate)
        assertEquals(2, callCount, "builder must be called twice: GPU attempt then CPU fallback")
    }

    @Test
    fun cpuOnly_activeDelegateIsCpu() {
        val lm = landmarker { _, _, delegate ->
            assertEquals(Delegate.CPU, delegate)
            null
        }
        lm.initialize(1, 0.5f, useGpu = false)
        assertEquals(Delegate.CPU, lm.activeDelegate)
    }

    @Test
    fun cpuFailure_rethrows() {
        val lm = landmarker { _, _, _ ->
            throw RuntimeException("CPU model load failed")
        }
        assertFailsWith<RuntimeException>("CPU failure must propagate") {
            lm.initialize(1, 0.5f, useGpu = false)
        }
    }
}
