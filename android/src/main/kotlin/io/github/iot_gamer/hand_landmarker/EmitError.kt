package io.github.iot_gamer.hand_landmarker

import io.flutter.plugin.common.EventChannel

/**
 * Pure-core error emission helper.
 * Android-import-free (no Handler/Looper) — JVM unit-testable.
 * Caller is responsible for invoking on the UI thread when required.
 */
internal fun emitError(sink: EventChannel.EventSink?, code: String, msg: String) =
    sink?.error(code, msg, null)
