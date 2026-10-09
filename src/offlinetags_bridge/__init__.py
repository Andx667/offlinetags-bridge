"""Local daemon and editor glue for OfflineTags privacy-badge detection (ONNX runtime).

Keep this module free of heavy imports: the client starts once per image and
must not pay for loading OpenCV or onnxruntime.
"""

__version__ = "0.1.0"
