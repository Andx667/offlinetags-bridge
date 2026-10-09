"""YOLOv8 detection on onnxruntime, without torch or ultralytics.

Expects a model exported by ``tools/export_onnx.py``: one input of shape
(1, 3, H, W) and one raw output of shape (1, 4 + classes, anchors), i.e. an
export without embedded NMS. With dynamic input axes the image is padded only
to the next multiple of the stride, like ultralytics does; a fixed-size model
gets a full square letterbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PAD_VALUE = 114
IMAGE_SIZE = 640
STRIDE = 32


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size`` (h, w).

    Returns the padded image, the scale gain and the (left, top) padding.
    """
    h, w = img.shape[:2]
    th, tw = size
    gain = min(th / h, tw / w)
    nh, nw = round(h * gain), round(w * gain)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = round((th - nh) / 2 - 0.1)
    left = round((tw - nw) / 2 - 0.1)
    out = np.full((th, tw, 3), PAD_VALUE, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, gain, (left, top)


def rect_size(shape: tuple[int, int]) -> tuple[int, int]:
    """Smallest stride-aligned (h, w) that holds the image scaled to IMAGE_SIZE."""
    h, w = shape
    gain = IMAGE_SIZE / max(h, w)
    return (
        -(-round(h * gain) // STRIDE) * STRIDE,
        -(-round(w * gain) // STRIDE) * STRIDE,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Plain non-maximum suppression over xyxy boxes. Returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode(
    output: np.ndarray,
    gain: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    conf: float,
    iou: float = 0.7,
    classes: list[int] | None = None,
) -> list[Box]:
    """Turn a raw YOLOv8 output tensor into boxes in original-image pixels."""
    pred = output[0].T  # (anchors, 4 + classes)
    scores = pred[:, 4:]
    if classes is not None:
        mask = np.zeros(scores.shape[1], dtype=bool)
        mask[classes] = True
        scores = np.where(mask, scores, 0.0)
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    keep = confidences > conf
    if not keep.any():
        return []
    xywh, confidences, class_ids = pred[keep, :4], confidences[keep], class_ids[keep]

    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    # shift each class into its own coordinate range so NMS never mixes classes
    offsets = class_ids[:, None].astype(boxes.dtype) * 7680.0
    kept = nms(boxes + offsets, confidences, iou)

    oh, ow = orig_shape
    boxes = boxes[kept]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad[0]) / gain, 0, ow)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad[1]) / gain, 0, oh)

    return [
        Box(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(confidences[i]), int(class_ids[i]))
        for b, i in zip(boxes, kept)
    ]


class YoloOnnx:
    """A YOLOv8 detection model loaded once and reused for every image."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.path = Path(model_path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        h, w = model_input.shape[2], model_input.shape[3]
        # None means dynamic axes: pick the input size per image
        self.size = (h, w) if isinstance(h, int) and isinstance(w, int) else None

    def predict(
        self,
        img_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        classes: list[int] | None = None,
    ) -> list[Box]:
        padded, gain, pad = letterbox(img_bgr, self.size or rect_size(img_bgr.shape[:2]))
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32)[None] / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        return decode(output, gain, pad, img_bgr.shape[:2], conf, iou, classes)
