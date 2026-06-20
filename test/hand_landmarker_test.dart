import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

void main() {
  group('parseLandmarks', () {
    test('empty list returns empty', () {
      expect(parseLandmarks([]), isEmpty);
    });

    test('one Float32List(63) returns 1 Hand with 21 landmarks', () {
      final data = Float32List.fromList(List.generate(63, (i) => i * 0.01));
      final hands = parseLandmarks([data]);
      expect(hands.length, 1);
      expect(hands[0].landmarks.length, 21);
      // x=data[0], y=data[1], z=data[2] for landmark 0
      expect(hands[0].landmarks[0].x, closeTo(0.00, 1e-6));
      expect(hands[0].landmarks[0].y, closeTo(0.01, 1e-6));
      expect(hands[0].landmarks[0].z, closeTo(0.02, 1e-6));
      // landmark 1: indices 3,4,5
      expect(hands[0].landmarks[1].x, closeTo(0.03, 1e-6));
      expect(hands[0].landmarks[1].y, closeTo(0.04, 1e-6));
      expect(hands[0].landmarks[1].z, closeTo(0.05, 1e-6));
    });

    test('two Float32List(63) returns 2 Hands', () {
      final hands = parseLandmarks([Float32List(63), Float32List(63)]);
      expect(hands.length, 2);
      expect(hands[0].landmarks.length, 21);
      expect(hands[1].landmarks.length, 21);
    });

    test('empty inner Float32List(0) returns Hand with 0 landmarks, no throw', () {
      final hands = parseLandmarks([Float32List(0)]);
      expect(hands.length, 1);
      expect(hands[0].landmarks, isEmpty);
    });
  });

  group('landmarkStream error propagation', () {
    test('stream errors surface via onError; normal events yield Hand', () async {
      final controller = StreamController<dynamic>.broadcast();
      final plugin = HandLandmarkerPlugin.testOnly(streamSource: controller.stream);

      final errors = <Object>[];
      final hands = <List<Hand>>[];

      final sub = plugin.landmarkStream.listen(
        hands.add,
        onError: errors.add,
      );

      controller.add([Float32List(63)]);
      await Future.microtask(() {});

      controller.addError(Exception('mediapipe_error'));
      await Future.microtask(() {});

      await sub.cancel();
      await controller.close();

      expect(hands.length, 1);
      expect(hands[0].length, 1);
      expect(errors.length, 1);
    });
  });

  group('dispose ordering', () {
    test('close() is called before release()', () {
      final fake = _FakeNativeLandmarker();
      final plugin = HandLandmarkerPlugin.testOnly(landmarker: fake);
      plugin.dispose();
      expect(fake.callOrder, ['close', 'release']);
    });
  });

  group('activeDelegate surfacing', () {
    test('plugin.activeDelegate returns cpu when fake stubs cpu', () {
      final fake = _FakeNativeLandmarker()
        ..stubbedDelegate = HandLandmarkerDelegate.cpu;
      final plugin = HandLandmarkerPlugin.testOnly(landmarker: fake);
      expect(plugin.activeDelegate, HandLandmarkerDelegate.cpu);
    });

    test('plugin.activeDelegate returns gpu when fake stubs gpu', () {
      final fake = _FakeNativeLandmarker()
        ..stubbedDelegate = HandLandmarkerDelegate.gpu;
      final plugin = HandLandmarkerPlugin.testOnly(landmarker: fake);
      expect(plugin.activeDelegate, HandLandmarkerDelegate.gpu);
    });
  });

  group('ByteBufferPool — no per-frame fromList', () {
    test('processFrame N times triggers allocate once per plane, fromList 0 times', () {
      final pool = _FakeByteBufferPool();
      final plugin = HandLandmarkerPlugin.testOnly(pool: pool);

      const n = 5;
      for (var i = 0; i < n; i++) {
        plugin.processFrameRaw(
          yBytes: Uint8List(100),
          uBytes: Uint8List(50),
          vBytes: Uint8List(50),
          width: 10,
          height: 10,
          yRowStride: 10,
          uvRowStride: 10,
          uvPixelStride: 1,
          rotation: 0,
          timestampMs: i,
        );
      }

      expect(pool.fromListCount, 0);
      // Each plane allocated once (no growth since sizes are stable)
      expect(pool.allocateCount, 3);
    });

    test('pool re-allocates when plane bytes grow', () {
      final pool = _FakeByteBufferPool();
      final plugin = HandLandmarkerPlugin.testOnly(pool: pool);

      plugin.processFrameRaw(
        yBytes: Uint8List(100),
        uBytes: Uint8List(50),
        vBytes: Uint8List(50),
        width: 10,
        height: 10,
        yRowStride: 10,
        uvRowStride: 10,
        uvPixelStride: 1,
        rotation: 0,
        timestampMs: 0,
      );

      plugin.processFrameRaw(
        yBytes: Uint8List(200),
        uBytes: Uint8List(100),
        vBytes: Uint8List(100),
        width: 20,
        height: 10,
        yRowStride: 20,
        uvRowStride: 20,
        uvPixelStride: 1,
        rotation: 0,
        timestampMs: 1,
      );

      expect(pool.fromListCount, 0);
      // 3 initial + 3 re-alloc on growth
      expect(pool.allocateCount, 6);
    });
  });

  group('Landmark model', () {
    test('holds correct values', () {
      final l = Landmark(0.1, 0.2, 0.3);
      expect(l.x, 0.1);
      expect(l.y, 0.2);
      expect(l.z, 0.3);
    });
  });

  group('MonotonicCounter (timestamp source)', () {
    test('next() yields strictly increasing 1,2,3', () {
      final c = MonotonicCounter();
      expect([c.next(), c.next(), c.next()], [1, 2, 3]);
    });
    test('two instances are independent', () {
      final a = MonotonicCounter(), b = MonotonicCounter();
      a.next(); a.next();
      expect(b.next(), 1);
    });
  });
}

// --- Fakes ---

class _FakeNativeLandmarker implements NativeLandmarker {
  final callOrder = <String>[];
  HandLandmarkerDelegate stubbedDelegate = HandLandmarkerDelegate.cpu;

  @override
  void close() => callOrder.add('close');

  @override
  void release() => callOrder.add('release');

  @override
  HandLandmarkerDelegate get activeDelegate => stubbedDelegate;
}

class _FakeBuffer implements ManagedByteBuffer {
  final Uint8List _view;

  _FakeBuffer(int size) : _view = Uint8List(size);

  @override
  Uint8List get view => _view;

  @override
  void setRange(int start, int end, Uint8List bytes) {
    _view.setRange(start, end, bytes);
  }

  @override
  void release() {}
}

class _FakeByteBufferPool implements ByteBufferPool {
  int allocateCount = 0;
  int fromListCount = 0;

  @override
  ManagedByteBuffer allocateDirect(int capacity) {
    allocateCount++;
    return _FakeBuffer(capacity);
  }

  @override
  ManagedByteBuffer fromList(Uint8List bytes) {
    fromListCount++;
    return _FakeBuffer(bytes.length);
  }
}
