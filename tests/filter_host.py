"""A minimal Photoshop filter host in ctypes, enough to drive the Affinity plug-in in tests.

FilterRecord is declared here independently of affinity/filter_api.h so the
two definitions check each other.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int16, c_int32, c_ubyte, c_uint16, c_uint32, c_void_p

import numpy as np

SELECTOR_PARAMETERS, SELECTOR_PREPARE, SELECTOR_START, SELECTOR_CONTINUE, SELECTOR_FINISH = 1, 2, 3, 4, 5
MODE_RGB, MODE_RGB48 = 3, 11
NO_ERR, USER_CANCELED = 0, -128


class Point16(ctypes.Structure):
    _fields_ = [("v", c_int16), ("h", c_int16)]


class Rect16(ctypes.Structure):
    _fields_ = [("top", c_int16), ("left", c_int16), ("bottom", c_int16), ("right", c_int16)]

    def is_empty(self) -> bool:
        return self.bottom <= self.top or self.right <= self.left


def _shorts(*names):
    return [(name, c_int16) for name in names]


def _pointers(*names):
    return [(name, c_void_p) for name in names]


class FilterRecord(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("serial", c_int32),
        *_pointers("abortProc", "progressProc", "parameters"),
        ("imageSize", Point16),
        ("planes", c_int16),
        ("filterRect", Rect16),
        ("background", c_uint16 * 3),
        ("foreground", c_uint16 * 3),
        ("maxSpace", c_int32),
        ("bufferSpace", c_int32),
        ("inRect", Rect16),
        *_shorts("inLoPlane", "inHiPlane"),
        ("outRect", Rect16),
        *_shorts("outLoPlane", "outHiPlane"),
        ("inData", c_void_p),
        ("inRowBytes", c_int32),
        ("outData", c_void_p),
        ("outRowBytes", c_int32),
        ("isFloating", c_ubyte),
        ("haveMask", c_ubyte),
        ("autoMask", c_ubyte),
        ("maskRect", Rect16),
        ("maskData", c_void_p),
        ("maskRowBytes", c_int32),
        ("backColor", c_ubyte * 4),
        ("foreColor", c_ubyte * 4),
        ("hostSig", c_uint32),
        ("hostProcs", c_void_p),
        ("imageMode", c_int16),
        ("imageHRes", c_int32),
        ("imageVRes", c_int32),
        ("floatCoord", Point16),
        ("wholeSize", Point16),
        ("monitor", c_int32 * 10),
        *_pointers("platformData", "bufferProcs", "resourceProcs", "processEvent",
                   "displayPixels", "handleProcs"),
        ("supportsDummyChannels", c_ubyte),
        ("supportsAlternateLayouts", c_ubyte),
        *_shorts("wantLayout", "filterCase", "dummyPlaneValue"),
        *_pointers("premiereHook", "advanceState"),
        ("supportsAbsolute", c_ubyte),
        ("wantsAbsolute", c_ubyte),
        ("getPropertyObsolete", c_void_p),
        ("cannotUndo", c_ubyte),
        ("supportsPadding", c_ubyte),
        *_shorts("inputPadding", "outputPadding", "maskPadding"),
        ("samplingSupport", c_ubyte),
        ("reservedByte", c_ubyte),
        ("inputRate", c_int32),
        ("maskRate", c_int32),
        ("colorServices", c_void_p),
        *_shorts(
            "inLayerPlanes", "inTransparencyMask", "inLayerMasks", "inInvertedLayerMasks", "inNonLayerPlanes",
            "outLayerPlanes", "outTransparencyMask", "outLayerMasks", "outInvertedLayerMasks", "outNonLayerPlanes",
            "absLayerPlanes", "absTransparencyMask", "absLayerMasks", "absInvertedLayerMasks", "absNonLayerPlanes",
            "inPreDummyPlanes", "inPostDummyPlanes", "outPreDummyPlanes", "outPostDummyPlanes",
        ),
        ("inColumnBytes", c_int32),
        ("inPlaneBytes", c_int32),
        ("outColumnBytes", c_int32),
        ("outPlaneBytes", c_int32),
        *_pointers("imageServicesProcs", "propertyProcs"),
        *_shorts("inTileHeight", "inTileWidth"), ("inTileOrigin", Point16),
        *_shorts("absTileHeight", "absTileWidth"), ("absTileOrigin", Point16),
        *_shorts("outTileHeight", "outTileWidth"), ("outTileOrigin", Point16),
        *_shorts("maskTileHeight", "maskTileWidth"), ("maskTileOrigin", Point16),
        *_pointers("descriptorParameters", "errorString", "channelPortProcs", "documentInfo",
                   "sSPBasic", "plugInRef"),
        ("depth", c_int32),
        ("iccProfileData", c_void_p),
        ("iccProfileSize", c_int32),
        ("canUseICCProfiles", c_int32),
        ("reserved", c_ubyte * 54),
    ]


def run_filter(plugin_path: str, rgb: np.ndarray) -> tuple[int, np.ndarray]:
    """Run the plug-in on an RGB image (uint8, or uint16 in Photoshop's 0..32768 range).

    Uses the classic protocol: Start announces the rectangles, the host fills
    the buffers, Continue does the work. Returns (result code, output image).
    """
    dll = ctypes.CDLL(plugin_path)
    entry = dll.PluginMain
    entry.restype = None
    entry.argtypes = [c_int16, ctypes.POINTER(FilterRecord), ctypes.POINTER(ctypes.c_ssize_t),
                      ctypes.POINTER(c_int16)]

    h, w = rgb.shape[:2]
    rec = FilterRecord()
    rec.imageSize = Point16(h, w)
    rec.wholeSize = Point16(h, w)
    rec.planes = 3
    rec.filterRect = Rect16(0, 0, h, w)
    rec.imageMode = MODE_RGB48 if rgb.dtype == np.uint16 else MODE_RGB
    rec.depth = 16 if rgb.dtype == np.uint16 else 8
    rec.filterCase = 1
    rec.maxSpace = 1 << 30

    data, result = ctypes.c_ssize_t(0), c_int16(0)
    source = np.ascontiguousarray(rgb)
    output = source.copy()

    for selector in (SELECTOR_PARAMETERS, SELECTOR_PREPARE, SELECTOR_START):
        entry(selector, ctypes.byref(rec), ctypes.byref(data), ctypes.byref(result))
        if result.value != NO_ERR:
            return result.value, output

    while not rec.inRect.is_empty() or not rec.outRect.is_empty():
        r = rec.inRect
        assert (r.top, r.left, r.bottom, r.right) == (0, 0, h, w), "host only serves the whole image"
        assert (rec.inLoPlane, rec.inHiPlane, rec.outLoPlane, rec.outHiPlane) == (0, 2, 0, 2)
        # garbage in the output buffer shows whether the plug-in fills all of it
        scratch = np.full_like(source, 77)
        rec.inData = source.ctypes.data
        rec.outData = scratch.ctypes.data
        rec.inRowBytes = rec.outRowBytes = source.strides[0]
        entry(SELECTOR_CONTINUE, ctypes.byref(rec), ctypes.byref(data), ctypes.byref(result))
        if result.value != NO_ERR:
            return result.value, output
        output = scratch

    entry(SELECTOR_FINISH, ctypes.byref(rec), ctypes.byref(data), ctypes.byref(result))
    return result.value, output
