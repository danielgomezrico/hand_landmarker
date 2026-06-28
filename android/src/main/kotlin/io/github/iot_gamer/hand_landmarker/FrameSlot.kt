package io.github.iot_gamer.hand_landmarker

/**
 * Worker-owned buffer holder for one captured camera frame.
 * Android-free — byte arrays are sized/resized by the producer before each copy.
 * Pooled via [FrameDrainCoordinator.pollFree]/[FrameDrainCoordinator.offerFree]
 * so object allocation is amortized across frames.
 */
internal class FrameSlot {
    var yArr: ByteArray = ByteArray(0)
    var uArr: ByteArray = ByteArray(0)
    var vArr: ByteArray = ByteArray(0)
    var width: Int = 0
    var height: Int = 0
    var uvRowStride: Int = 0
    var uvPixelStride: Int = 0
    var rotation: Int = 0
    var timestampMs: Long = 0L
}
