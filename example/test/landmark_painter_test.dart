import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import 'package:hand_landmarker_example/main.dart';

// A minimal Canvas implementation that counts geometry draw ops.
// Canvas cannot be extended (abstract with factory ctors), so we implement it.
class _CountingCanvas implements Canvas {
  int drawCircleCount = 0;
  int drawLineCount = 0;

  @override
  void drawCircle(Offset c, double radius, Paint paint) => drawCircleCount++;

  @override
  void drawLine(Offset p1, Offset p2, Paint paint) => drawLineCount++;

  // --- required Canvas stubs (never called by LandmarkPainter) ---
  @override
  void clipPath(Path path, {bool doAntiAlias = true}) {}
  @override
  void clipRRect(RRect rrect, {bool doAntiAlias = true}) {}
  @override
  void clipRect(Rect rect,
      {ui.ClipOp clipOp = ui.ClipOp.intersect, bool doAntiAlias = true}) {}
  @override
  void drawArc(Rect rect, double startAngle, double sweepAngle,
      bool useCenter, Paint paint) {}
  @override
  void drawAtlas(ui.Image atlas, List<RSTransform> transforms, List<Rect> rects,
      List<Color>? colors, BlendMode? blendMode, Rect? cullRect, Paint paint) {}
  @override
  void drawColor(Color color, BlendMode blendMode) {}
  @override
  void drawDRRect(RRect outer, RRect inner, Paint paint) {}
  @override
  void drawImage(ui.Image image, Offset offset, Paint paint) {}
  @override
  void drawImageNine(
      ui.Image image, Rect center, Rect dst, Paint paint) {}
  @override
  void drawImageRect(
      ui.Image image, Rect src, Rect dst, Paint paint) {}
  @override
  void drawOval(Rect rect, Paint paint) {}
  @override
  void drawPaint(Paint paint) {}
  @override
  void drawParagraph(ui.Paragraph paragraph, Offset offset) {}
  @override
  void drawPath(Path path, Paint paint) {}
  @override
  void drawPicture(ui.Picture picture) {}
  @override
  void drawPoints(
      ui.PointMode pointMode, List<Offset> points, Paint paint) {}
  @override
  void drawRRect(RRect rrect, Paint paint) {}
  @override
  void drawRawAtlas(
      ui.Image atlas,
      Float32List rstTransforms,
      Float32List rects,
      Int32List? colors,
      BlendMode? blendMode,
      Rect? cullRect,
      Paint paint) {}
  @override
  void drawRawPoints(
      ui.PointMode pointMode, Float32List points, Paint paint) {}
  @override
  void drawRect(Rect rect, Paint paint) {}
  @override
  void drawShadow(
      Path path, Color color, double elevation, bool transparentOccluder) {}
  @override
  void drawVertices(
      ui.Vertices vertices, BlendMode blendMode, Paint paint) {}
  @override
  int getSaveCount() => 1;
  @override
  void restore() {}
  @override
  void restoreToCount(int count) {}
  @override
  void rotate(double radians) {}
  @override
  void save() {}
  @override
  void saveLayer(Rect? bounds, Paint paint) {}
  @override
  void scale(double sx, [double? sy]) {}
  @override
  void skew(double sx, double sy) {}
  @override
  void transform(Float64List matrix4) {}
  @override
  void translate(double dx, double dy) {}

  @override
  ui.Rect getLocalClipBounds() => ui.Rect.zero;
  @override
  ui.Rect getDestinationClipBounds() => ui.Rect.zero;
  @override
  Float64List getTransform() => Float64List(16);
  @override
  void clipRSuperellipse(RSuperellipse rse, {bool doAntiAlias = true}) {}
  @override
  void drawRSuperellipse(RSuperellipse rse, Paint paint) {}
}

Hand _hand(List<List<double>> coords) => Hand(
      coords.map((c) => Landmark(c[0], c[1], c[2])).toList(),
    );

// A hand with 21 landmarks at distinct positions offset by [offset].
Hand _distinctHand(double offset) => _hand(
      List.generate(21, (i) => [i * 0.01 + offset, i * 0.01 + offset, 0.0]),
    );

LandmarkPainter _painter(
  List<Hand> hands, {
  Size previewSize = const Size(640, 480),
  CameraLensDirection lensDirection = CameraLensDirection.back,
  int sensorOrientation = 90,
}) =>
    LandmarkPainter(
      hands: hands,
      previewSize: previewSize,
      lensDirection: lensDirection,
      sensorOrientation: sensorOrientation,
    );

void main() {
  group('LandmarkPainter.shouldRepaint matrix', () {
    test('empty↔empty returns false (no repaint on steady no-hand scene)', () {
      final old = _painter([]);
      final current = _painter([]);
      expect(current.shouldRepaint(old), isFalse);
    });

    test('same-content fresh lists return false (content compare, not reference)', () {
      final old = _painter([_distinctHand(0.0)]);
      // Structurally equal hand, new object.
      final current = _painter([_distinctHand(0.0)]);
      expect(current.shouldRepaint(old), isFalse);
    });

    test('count 0↔1 returns true', () {
      final old = _painter([]);
      final current = _painter([_distinctHand(0.0)]);
      expect(current.shouldRepaint(old), isTrue);
    });

    test('sensorOrientation change returns true', () {
      final old = _painter([], sensorOrientation: 90);
      final current = _painter([], sensorOrientation: 0);
      expect(current.shouldRepaint(old), isTrue);
    });

    test('lensDirection change returns true', () {
      final old = _painter([], lensDirection: CameraLensDirection.back);
      final current = _painter([], lensDirection: CameraLensDirection.front);
      expect(current.shouldRepaint(old), isTrue);
    });

    test('previewSize change returns true', () {
      final old = _painter([], previewSize: const Size(640, 480));
      final current = _painter([], previewSize: const Size(1280, 720));
      expect(current.shouldRepaint(old), isTrue);
    });

    test('different landmark coords returns true', () {
      final old = _painter([_distinctHand(0.0)]);
      final current = _painter([_distinctHand(0.5)]);
      expect(current.shouldRepaint(old), isTrue);
    });
  });

  group('LandmarkPainter.paint — phantom-landmark guard', () {
    test('draws zero geometry when hands is empty', () {
      final canvas = _CountingCanvas();
      _painter([]).paint(canvas, const Size(300, 400));

      expect(canvas.drawCircleCount, 0);
      expect(canvas.drawLineCount, 0);
    });

    test('draws geometry when a hand is present', () {
      final canvas = _CountingCanvas();
      _painter([_distinctHand(0.0)]).paint(canvas, const Size(300, 400));

      // 21 landmarks → 21 circles; connections → matching line count
      expect(canvas.drawCircleCount, 21);
      expect(canvas.drawLineCount, HandLandmarkConnections.connections.length);
    });
  });
}
