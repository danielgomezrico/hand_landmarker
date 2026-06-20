package io.github.iot_gamer.hand_landmarker

/**
 * Pure monotonic-timestamp decision gate.
 * Android-import-free — JVM unit-testable.
 * CAS drop form (accumulateAndGet-max is BANNED per W1/plan).
 */
internal object TimestampGate {
    fun shouldProcess(incoming: Long, last: Long): Boolean = incoming > last
}
