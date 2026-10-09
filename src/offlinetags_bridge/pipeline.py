"""Detect privacy badges and apply each person's preference.

A port of the ``offlinetags`` 1.0.2 pipeline (MIT, github.com/fiker31/Offline-Tags)
onto ONNX models. Smart batch (cross-image face matching) is not ported.

Deliberate differences from upstream:
- the mosaic gets coarser on large faces (see Config.max_face_blocks); upstream's
  fixed 25 px blocks leave a face in a 20 MP photo recognisable
- annotation colours are real BGR (upstream passes RGB, so "Upload Me" came out blue)
- annotation line width and label size scale with the image
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .paths import PERSON_MODEL, TAG_MODEL, models_dir
from .yolo import YoloOnnx

CLASS_MAP = {
    0: {"tag_type": "Blur Me",   "circle_color": "blue",   "inner_symbol": "minus"},
    1: {"tag_type": "No Photo",  "circle_color": "red",    "inner_symbol": "x"},
    2: {"tag_type": "Tag Me",    "circle_color": "green",  "inner_symbol": "target"},
    3: {"tag_type": "Upload Me", "circle_color": "yellow", "inner_symbol": "check"},
}

ANNOTATE_STYLES = {  # colours are BGR
    "Tag Me":    {"color": (94, 197, 34), "label": "Tag Me  OK"},
    "Upload Me": {"color": (8, 179, 234), "label": "Upload Me  OK"},
}

PERSON_CLASS = 0  # COCO

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Config:
    tag_conf_threshold: float = 0.25
    person_conf_threshold: float = 0.30
    face_padding: float = 0.45   # extra margin around a blurred face
    pixel_block: int = 25        # minimum mosaic block size in pixels
    max_face_blocks: int = 12    # at most this many mosaic blocks across a face


@dataclass
class Detection:
    tag_type: str
    circle_color: str
    inner_symbol: str
    confidence: float
    position: dict  # {"x", "y", "radius"} in pixels

    @property
    def box(self) -> list[int]:
        p = self.position
        return [p["x"] - p["radius"], p["y"] - p["radius"], p["x"] + p["radius"], p["y"] + p["radius"]]


@dataclass
class Result:
    action: str  # "clean" | "annotated" | "blurred" | "discarded"
    detections: list[Detection] = field(default_factory=list)
    # [x1, y1, x2, y2] of every pixelated face, so an editor can redo the blur itself
    blur_regions: list[list[int]] = field(default_factory=list)
    processed: np.ndarray | None = None  # BGR; only for "annotated" / "blurred"

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "detections": [asdict(d) for d in self.detections],
            "blur_regions": self.blur_regions,
        }


def _iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _pixelate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              pixel_block: int, max_blocks: int) -> None:
    """Mosaic an elliptical area in place."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    block = max(pixel_block, max(h, w) // max_blocks)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    img[y1:y2, x1:x2] = np.where(mask[..., None] > 0, pixelated, roi)


def _head_fallback(px1: int, py1: int, px2: int, py2: int) -> list[tuple[int, int, int, int]]:
    """No face found: assume the head is the top 22% of the person box."""
    ph, pw = py2 - py1, px2 - px1
    margin = int(pw * 0.15)
    return [(margin, 0, pw - 2 * margin, int(ph * 0.22))]


def _draw_label(img: np.ndarray, text: str, x1: int, y1: int, color: tuple, scale: float) -> None:
    font_scale, thick, pad = 0.7 * scale, round(2 * scale), round(6 * scale)
    (tw, th), _ = cv2.getTextSize(text, _FONT, font_scale, thick)
    y1 = max(y1, th + pad * 2)  # keep the label inside the image
    cv2.rectangle(img, (x1, y1 - th - pad * 2), (x1 + tw + pad * 2, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), _FONT, font_scale, (0, 0, 0), thick, cv2.LINE_AA)


class Pipeline:
    """Loads both models once; ``process`` is safe to call from several threads."""

    def __init__(self, model_dir: str | Path | None = None, config: Config | None = None) -> None:
        self.config = config or Config()
        self.model_dir = Path(model_dir) if model_dir else models_dir()
        for name in (TAG_MODEL, PERSON_MODEL):
            if not (self.model_dir / name).exists():
                raise FileNotFoundError(
                    f"{name} not found in {self.model_dir}. "
                    "Create the models with tools/export_onnx.py (see README)."
                )
        self.tag_model = YoloOnnx(self.model_dir / TAG_MODEL)
        self.person_model = YoloOnnx(self.model_dir / PERSON_MODEL)
        self._cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + name)
            for name in (
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_frontalface_alt_tree.xml",
            )
        ]
        # Haar cascades are not thread-safe, so one image at a time
        self._lock = threading.Lock()

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        # the tag model was trained on grayscale images stored as 3 channels
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        detections = []
        for box in self.tag_model.predict(gray3, conf=self.config.tag_conf_threshold):
            meta = CLASS_MAP.get(box.class_id, {})
            detections.append(Detection(
                tag_type=meta.get("tag_type", "Unknown"),
                circle_color=meta.get("circle_color", ""),
                inner_symbol=meta.get("inner_symbol", ""),
                confidence=round(box.confidence, 3),
                position={
                    "x": (box.x1 + box.x2) // 2,
                    "y": (box.y1 + box.y2) // 2,
                    "radius": (box.x2 - box.x1) // 2,
                },
            ))
        return detections

    def process(self, img_bgr: np.ndarray, annotate: bool = True) -> Result:
        """Detect tags in one BGR image and apply the matching action."""
        with self._lock:
            detections = self.detect(img_bgr)
            tag_types = {d.tag_type for d in detections}

            if "No Photo" in tag_types:
                return Result("discarded", detections)

            wants_blur = "Blur Me" in tag_types
            wants_annotate = annotate and bool(tag_types & ANNOTATE_STYLES.keys())
            if not (wants_blur or wants_annotate):
                return Result("clean", detections)

            persons = [
                [b.x1, b.y1, b.x2, b.y2]
                for b in self.person_model.predict(
                    img_bgr, conf=self.config.person_conf_threshold, classes=[PERSON_CLASS])
            ]
            out = img_bgr.copy()
            regions: list[list[int]] = []
            if wants_blur:
                regions = self._blur(img_bgr, out, detections, persons)
            if wants_annotate:
                self._annotate(out, detections, persons)
            return Result("annotated" if wants_annotate else "blurred", detections, regions, out)

    def _find_faces(self, img: np.ndarray, px1: int, py1: int, px2: int, py2: int, top_frac: float):
        """Search the top part of a person box with each cascade; first hit wins."""
        search_y2 = min(py2, py1 + int((py2 - py1) * top_frac))
        region = img[py1:search_y2, px1:px2]
        if region.size == 0:
            return []
        scale = 1.0
        fh, fw = region.shape[:2]
        if fw > 800:
            scale = 800 / fw
            region = cv2.resize(region, (int(fw * scale), int(fh * scale)))
        gray = cv2.equalizeHist(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
        for cascade in self._cascades:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15))
            if len(faces) > 0:
                return [tuple(int(v / scale) for v in face) for face in faces]
        return []

    def _blur(self, img: np.ndarray, out: np.ndarray, detections: list[Detection],
              persons: list[list[int]]) -> list[list[int]]:
        ih, iw = img.shape[:2]
        pad = self.config.face_padding
        regions = []
        for det in detections:
            if det.tag_type != "Blur Me":
                continue
            tag_box = det.box
            if persons:
                px1, py1, px2, py2 = max(persons, key=lambda pb: _iou(tag_box, pb))
            else:
                # no person found: search the area above the badge
                half_w = iw // 4
                px1, py1 = max(0, det.position["x"] - half_w), 0
                px2, py2 = min(iw, det.position["x"] + half_w), det.position["y"]

            faces = (self._find_faces(img, px1, py1, px2, py2, 0.35)
                     or self._find_faces(img, px1, py1, px2, py2, 0.50)
                     or _head_fallback(px1, py1, px2, py2))

            for fx, fy, fw, fh in faces:
                pad_x, pad_y = int(fw * pad), int(fh * pad)
                x1, y1 = max(0, px1 + fx - pad_x), max(0, py1 + fy - pad_y)
                x2, y2 = min(iw, px1 + fx + fw + pad_x), min(ih, py1 + fy + fh + pad_y)
                _pixelate(out, x1, y1, x2, y2, self.config.pixel_block, self.config.max_face_blocks)
                regions.append([x1, y1, x2, y2])
        return regions

    def _annotate(self, out: np.ndarray, detections: list[Detection], persons: list[list[int]]) -> None:
        ih, iw = out.shape[:2]
        scale = max(1.0, max(ih, iw) / 1000)  # upstream's sizes suit a ~1000 px image
        for det in detections:
            style = ANNOTATE_STYLES.get(det.tag_type)
            if not style:
                continue
            if persons:
                px1, py1, px2, py2 = max(persons, key=lambda pb: _iou(det.box, pb))
            else:
                p, r = det.position, det.position["radius"] + 20
                px1, py1 = max(0, p["x"] - r), max(0, p["y"] - r)
                px2, py2 = min(iw, p["x"] + r), min(ih, p["y"] + r)
            cv2.rectangle(out, (px1, py1), (px2, py2), style["color"], round(4 * scale))
            _draw_label(out, style["label"], px1, py1, style["color"], scale)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR. Goes through numpy so non-ASCII Windows paths work."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray, jpeg_quality: int = 92) -> None:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError(f"Could not encode image as {path.suffix}")
    buf.tofile(str(path))
