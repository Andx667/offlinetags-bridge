"""Export the OfflineTags models to ONNX. Run once, in an environment with
tools/requirements-export.txt installed; the runtime never needs torch.

    python tools/export_onnx.py              # writes models/tags.onnx and models/person.onnx

The tag model is taken from the ``offlinetags`` wheel on PyPI (checked against
PyPI's SHA-256) unless --tag-weights points at a local .pt file. The person
model is the stock ``yolov8n.pt``, which ultralytics downloads itself.

Loading a .pt file unpickles it, which can run code. Only export weights you trust.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM_VERSION = "1.0.2"
WHEEL_MEMBER = "offlinetags/models/train_gray_best.pt"


def fetch_tag_weights(version: str, dest: Path) -> Path:
    with urllib.request.urlopen(f"https://pypi.org/pypi/offlinetags/{version}/json") as response:
        files = json.load(response)["urls"]
    wheel = next(f for f in files if f["filename"].endswith(".whl"))
    print(f"downloading {wheel['filename']} ...")
    with urllib.request.urlopen(wheel["url"]) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
        raise SystemExit("SHA-256 of the downloaded wheel does not match PyPI")
    target = dest / "train_gray_best.pt"
    target.write_bytes(zipfile.ZipFile(io.BytesIO(data)).read(WHEEL_MEMBER))
    return target


def export(weights: Path | str, target: Path, imgsz: int, opset: int | None) -> None:
    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, target)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--tag-weights", type=Path, help="local train_gray_best.pt instead of the PyPI wheel")
    parser.add_argument("--person-weights", default="yolov8n.pt")
    parser.add_argument("--upstream-version", default=UPSTREAM_VERSION)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        tag_weights = args.tag_weights or fetch_tag_weights(args.upstream_version, work)
        export(tag_weights, args.out / "tags.onnx", args.imgsz, args.opset)

        person = Path(args.person_weights)
        if not person.exists():
            # let ultralytics download into the temp directory, not the current one
            person = work / person.name
        export(person, args.out / "person.onnx", args.imgsz, args.opset)


if __name__ == "__main__":
    main()
