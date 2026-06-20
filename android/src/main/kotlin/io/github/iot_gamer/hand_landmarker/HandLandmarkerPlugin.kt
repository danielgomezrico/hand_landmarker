package io.github.iot_gamer.hand_landmarker

import android.os.Handler
import android.os.Looper
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

class HandLandmarkerPlugin : FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler,
    HandLandmarkListener {

    private lateinit var methodChannel: MethodChannel
    private lateinit var eventChannel: EventChannel

    private var eventSink: EventChannel.EventSink? = null
    private val uiThreadHandler = Handler(Looper.getMainLooper())

    companion object {
        // Bridge across the JNI/engine instance split: Dart constructs MyHandLandmarker
        // via JNI and cannot see this engine-registered plugin. The plugin publishes
        // itself here at engine attach (which happens before Dart's create()), so the
        // JNI landmarker can pull it as its result listener. @Volatile: written on the
        // platform thread, read on the JNI/init thread.
        @Volatile
        internal var activeListener: HandLandmarkListener? = null
    }

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel = MethodChannel(flutterPluginBinding.binaryMessenger, "hand_landmarker")
        methodChannel.setMethodCallHandler(this)

        eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, "hand_landmarker/events")
        eventChannel.setStreamHandler(this)

        activeListener = this
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
    }

    // K4 — HandLandmarkListener: receives binary payload from MyHandLandmarker
    override fun onLandmarksDetected(hands: List<FloatArray>) {
        uiThreadHandler.post {
            eventSink?.success(hands)
        }
    }

    // K4 — HandLandmarkListener: receives errors from MyHandLandmarker
    override fun onError(code: String, message: String) {
        uiThreadHandler.post {
            emitError(eventSink, code, message)
        }
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        result.notImplemented()
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        eventSink = null
        if (activeListener === this) activeListener = null
    }
}
