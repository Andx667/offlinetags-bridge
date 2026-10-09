import numpy as np

from offlinetags_bridge.yolo import PAD_VALUE, decode, letterbox, nms, rect_size


def test_letterbox_pads_wide_image_top_and_bottom():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, gain, (left, top) = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert gain == 3.2
    assert (left, top) == (0, 160)
    assert out[0, 0, 0] == PAD_VALUE and out[320, 320, 0] == 0


def test_rect_size_pads_only_to_the_stride():
    assert rect_size((4000, 6000)) == (448, 640)   # 427 -> 448
    assert rect_size((6000, 4000)) == (640, 448)
    assert rect_size((100, 100)) == (640, 640)


def test_nms_drops_overlapping_lower_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def _raw_output(rows, classes=4, anchors=20):
    """Build a (1, 4 + classes, anchors) tensor from (cx, cy, w, h, class, score) rows."""
    out = np.zeros((1, 4 + classes, anchors), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_decode_maps_boxes_back_to_original_pixels():
    # 200x100 image letterboxed to 640: gain 3.2, 160 px padding on top
    output = _raw_output([(320, 320, 64, 64, 2, 0.9)])
    boxes = decode(output, 3.2, (0, 160), (100, 200), conf=0.25)
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (90, 40, 110, 60)
    assert b.class_id == 2 and abs(b.confidence - 0.9) < 1e-6


def test_decode_filters_confidence_and_keeps_overlapping_boxes_of_other_classes():
    output = _raw_output([
        (100, 100, 40, 40, 0, 0.9),
        (101, 101, 40, 40, 0, 0.6),   # same class, overlapping: suppressed
        (100, 100, 40, 40, 1, 0.5),   # other class, overlapping: kept
        (300, 300, 40, 40, 3, 0.1),   # below threshold
    ])
    boxes = decode(output, 1.0, (0, 0), (640, 640), conf=0.25)
    assert sorted((b.class_id, round(b.confidence, 1)) for b in boxes) == [(0, 0.9), (1, 0.5)]


def test_decode_class_filter():
    output = _raw_output([(100, 100, 40, 40, 0, 0.9), (300, 300, 40, 40, 1, 0.9)])
    boxes = decode(output, 1.0, (0, 0), (640, 640), conf=0.25, classes=[0])
    assert [b.class_id for b in boxes] == [0]


def test_decode_empty():
    assert decode(_raw_output([]), 1.0, (0, 0), (640, 640), conf=0.25) == []
