import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:jni/jni.dart';
import 'package:jni_flutter/jni_flutter.dart';

import 'hand_landmarker_bindings.dart';

// --- Public Data Models ---

/// A detected hand with its landmarks.
class Hand {
  final List<Landmark> landmarks;
  Hand(this.landmarks);
}

/// A single landmark point with its 3D coordinates.
class Landmark {
  final double x;
  final double y;
  final double z;
  Landmark(this.x, this.y, this.z);
}

enum HandLandmarkerDelegate { cpu, gpu }

/// Decodes the binary wire payload from the EventChannel.
/// Each element of [event] is a [Float32List] with handSize*3 floats (x,y,z interleaved).
List<Hand> parseLandmarks(List<dynamic> event) {
  final hands = <Hand>[];
  for (final element in event) {
    final floats = element as Float32List;
    final landmarks = <Landmark>[];
    for (var i = 0; i + 2 < floats.length; i += 3) {
      landmarks.add(Landmark(floats[i].toDouble(), floats[i + 1].toDouble(), floats[i + 2].toDouble()));
    }
    hands.add(Hand(landmarks));
  }
  return hands;
}

// --- Seam interfaces (injectable for testing) ---

/// Wraps a JNI direct ByteBuffer with a cached [Uint8List] view.
abstract class ManagedByteBuffer {
  Uint8List get view;
  void setRange(int start, int end, Uint8List bytes);
  void release();
}

/// Allocates JNI ByteBuffers. Default impl uses [JByteBuffer]; fakes record calls.
abstract class ByteBufferPool {
  ManagedByteBuffer allocateDirect(int capacity);
  ManagedByteBuffer fromList(Uint8List bytes);
}

/// Thin interface over the JNI landmarker — enables dispose-order testing.
abstract class NativeLandmarker {
  void close();
  void release();
  HandLandmarkerDelegate get activeDelegate;
}

// --- Default production implementations ---

class _RealBuffer implements ManagedByteBuffer {
  final JByteBuffer _buf;
  final Uint8List _view;

  _RealBuffer._fromBuf(JByteBuffer buf)
      : _buf = buf,
        _view = buf.asUint8List();

  @override
  Uint8List get view => _view;

  @override
  void setRange(int start, int end, Uint8List bytes) {
    _view.setRange(start, end, bytes);
  }

  @override
  void release() => _buf.release();

  JByteBuffer get jBuffer => _buf;
}

class _RealByteBufferPool implements ByteBufferPool {
  @override
  ManagedByteBuffer allocateDirect(int capacity) {
    final buf = JByteBuffer.allocateDirect(capacity);
    return _RealBuffer._fromBuf(buf);
  }

  @override
  ManagedByteBuffer fromList(Uint8List bytes) => throw UnsupportedError(
      'fromList is never used in the hot path — use allocateDirect + setRange');
}

class _RealNativeLandmarker implements NativeLandmarker {
  final MyHandLandmarker _inner;
  _RealNativeLandmarker(this._inner);

  @override
  void close() => _inner.close();

  @override
  void release() => _inner.release();

  @override
  HandLandmarkerDelegate get activeDelegate {
    final jStr = _inner.getActiveDelegate();
    final name = jStr.toDartString(releaseOriginal: true);
    return name == 'GPU' ? HandLandmarkerDelegate.gpu : HandLandmarkerDelegate.cpu;
  }
}

// --- Timestamp seam ---

/// Source of strictly-increasing per-frame timestamps for LIVE_STREAM inference.
/// CameraImage exposes no usable monotonic per-frame timestamp in camera 0.12.0+1,
/// so the plugin generates one. Injectable for deterministic tests.
abstract class TimestampSource {
  int next();
}

/// Default [TimestampSource]: a per-instance monotonic counter (1, 2, 3, ...).
class MonotonicCounter implements TimestampSource {
  int _v = 0;
  @override
  int next() => ++_v;
}

// --- Plugin ---

/// The main class for the Hand Landmarker plugin.
class HandLandmarkerPlugin {
  static const EventChannel _eventChannel = EventChannel('hand_landmarker/events');

  final NativeLandmarker _landmarker;
  final ByteBufferPool _pool;
  final Stream<dynamic> _streamSource;
  final TimestampSource _timestampSource;

  Stream<List<Hand>>? _landmarkStream;

  // Cached JNI buffers — allocated once per plane, re-used each frame.
  ManagedByteBuffer? _yBuf;
  ManagedByteBuffer? _uBuf;
  ManagedByteBuffer? _vBuf;

  HandLandmarkerPlugin._(this._landmarker, this._pool, this._streamSource, this._timestampSource);

  /// Test-only constructor — injects fakes for any seam.
  @visibleForTesting
  HandLandmarkerPlugin.testOnly({
    NativeLandmarker? landmarker,
    ByteBufferPool? pool,
    Stream<dynamic>? streamSource,
    TimestampSource? timestampSource,
  }) : _landmarker = landmarker ?? _NoOpNativeLandmarker(),
       _pool = pool ?? _RealByteBufferPool(),
       _streamSource = streamSource ?? _eventChannel.receiveBroadcastStream(),
       _timestampSource = timestampSource ?? MonotonicCounter();

  /// Creates and initializes the Hand Landmarker.
  static HandLandmarkerPlugin create({
    int numHands = 2,
    double minHandDetectionConfidence = 0.5,
    HandLandmarkerDelegate delegate = HandLandmarkerDelegate.gpu,
  }) {
    final contextObj = androidApplicationContext;
    final landmarker = MyHandLandmarker(contextObj);
    landmarker.initialize(
      numHands,
      minHandDetectionConfidence,
      delegate == HandLandmarkerDelegate.gpu,
    );
    return HandLandmarkerPlugin._(
      _RealNativeLandmarker(landmarker),
      _RealByteBufferPool(),
      _eventChannel.receiveBroadcastStream(),
      MonotonicCounter(),
    );
  }

  /// Continuous stream of detected hand landmarks.
  Stream<List<Hand>> get landmarkStream {
    _landmarkStream ??= _streamSource.map((event) => parseLandmarks(event as List<dynamic>));
    return _landmarkStream!;
  }

  /// The delegate that is actually engaged after initialization.
  /// May differ from the requested delegate when GPU is unavailable and the
  /// plugin silently fell back to CPU.
  HandLandmarkerDelegate get activeDelegate => _landmarker.activeDelegate;

  /// Feeds a [CameraImage] into the native pipeline asynchronously.
  /// No JNI alloc per frame — buffers are cached and reused.
  void processFrame(CameraImage image, int sensorOrientation) {
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    processFrameRaw(
      yBytes: yPlane.bytes,
      uBytes: uPlane.bytes,
      vBytes: vPlane.bytes,
      width: image.width,
      height: image.height,
      yRowStride: yPlane.bytesPerRow,
      uvRowStride: uPlane.bytesPerRow,
      uvPixelStride: uPlane.bytesPerPixel!,
      rotation: sensorOrientation,
      timestampMs: _timestampSource.next(),
    );
  }

  /// Low-level frame submission — testable without [CameraImage].
  void processFrameRaw({
    required Uint8List yBytes,
    required Uint8List uBytes,
    required Uint8List vBytes,
    required int width,
    required int height,
    required int yRowStride,
    required int uvRowStride,
    required int uvPixelStride,
    required int rotation,
    required int timestampMs,
  }) {
    _yBuf = _ensureCapacity(_yBuf, yBytes);
    _uBuf = _ensureCapacity(_uBuf, uBytes);
    _vBuf = _ensureCapacity(_vBuf, vBytes);

    _yBuf!.setRange(0, yBytes.length, yBytes);
    _uBuf!.setRange(0, uBytes.length, uBytes);
    _vBuf!.setRange(0, vBytes.length, vBytes);

    // Only real pool returns a _RealBuffer with a JByteBuffer; fake pool used in tests.
    final yBuf = _yBuf!;
    final uBuf = _uBuf!;
    final vBuf = _vBuf!;

    if (yBuf is _RealBuffer && uBuf is _RealBuffer && vBuf is _RealBuffer) {
      (_landmarker as _RealNativeLandmarker)._inner.processFrame(
        yBuf.jBuffer,
        uBuf.jBuffer,
        vBuf.jBuffer,
        width,
        height,
        yRowStride,
        uvRowStride,
        uvPixelStride,
        rotation,
        timestampMs,
      );
    }
    // In tests the landmarker is a fake — no JNI call needed.
  }

  ManagedByteBuffer _ensureCapacity(ManagedByteBuffer? current, Uint8List bytes) {
    if (current == null || current.view.length < bytes.length) {
      current?.release();
      return _pool.allocateDirect(bytes.length);
    }
    return current;
  }

  /// Releases native resources. [close()] is called before [release()] per lifecycle contract.
  void dispose() {
    _landmarker.close();
    _landmarker.release();
    _yBuf?.release();
    _uBuf?.release();
    _vBuf?.release();
    _yBuf = null;
    _uBuf = null;
    _vBuf = null;
  }
}

/// No-op landmarker used when only the stream or pool seam is under test.
class _NoOpNativeLandmarker implements NativeLandmarker {
  @override
  void close() {}
  @override
  void release() {}
  @override
  HandLandmarkerDelegate get activeDelegate => HandLandmarkerDelegate.cpu;
}
