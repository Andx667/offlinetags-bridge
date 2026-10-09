/* The part of the Photoshop filter plug-in interface this plug-in uses.
 *
 * Written for this project so the build does not need the Adobe Photoshop SDK.
 * Field order and packing of FilterRecord follow the public interface as
 * mirrored by PSFilterPdn (MIT, github.com/0xC0000054/PSFilterPdn), a filter
 * host that loads real-world 64-bit plug-ins with this layout. Fields the
 * plug-in does not touch are kept as opaque pointers.
 */
#ifndef OFFLINETAGS_FILTER_API_H
#define OFFLINETAGS_FILTER_API_H

#include <stddef.h>
#include <stdint.h>

/* selectors passed to the entry point */
enum {
    filterSelectorAbout = 0,
    filterSelectorParameters = 1,
    filterSelectorPrepare = 2,
    filterSelectorStart = 3,
    filterSelectorContinue = 4,
    filterSelectorFinish = 5
};

/* result codes */
enum {
    noErr = 0,
    userCanceledErr = -128,
    filterBadParameters = -30100,
    filterBadMode = -30101
};

/* FilterRecord.imageMode */
enum {
    plugInModeRGBColor = 3,
    plugInModeRGB48 = 11
};

#pragma pack(push, 4)
#pragma warning(push)
#pragma warning(disable: 4121) /* packing changes member alignment: intended */

typedef struct { int16_t v, h; } Point16;
typedef struct { int16_t top, left, bottom, right; } Rect16;
typedef struct { uint16_t red, green, blue; } RGBColor16;
typedef struct {
    int32_t gamma, redX, redY, greenX, greenY, blueX, blueY, whiteX, whiteY, ambient;
} PlugInMonitor;

typedef struct FilterRecord {
    int32_t serial;
    void *abortProc;
    void *progressProc;
    void *parameters;
    Point16 imageSize;
    int16_t planes;
    Rect16 filterRect;          /* area to filter: the layer or the selection bounds */
    RGBColor16 background;
    RGBColor16 foreground;
    int32_t maxSpace;
    int32_t bufferSpace;
    Rect16 inRect;              /* plug-in sets: area it wants to read */
    int16_t inLoPlane;
    int16_t inHiPlane;
    Rect16 outRect;             /* plug-in sets: area it wants to write */
    int16_t outLoPlane;
    int16_t outHiPlane;
    void *inData;               /* host sets: interleaved samples of inRect */
    int32_t inRowBytes;
    void *outData;
    int32_t outRowBytes;
    uint8_t isFloating;
    uint8_t haveMask;
    uint8_t autoMask;
    Rect16 maskRect;
    void *maskData;
    int32_t maskRowBytes;
    uint8_t backColor[4];
    uint8_t foreColor[4];
    uint32_t hostSig;
    void *hostProcs;
    int16_t imageMode;
    int32_t imageHRes;
    int32_t imageVRes;
    Point16 floatCoord;
    Point16 wholeSize;
    PlugInMonitor monitor;
    void *platformData;
    void *bufferProcs;
    void *resourceProcs;
    void *processEvent;
    void *displayPixels;
    void *handleProcs;
    /* 3.0 */
    uint8_t supportsDummyChannels;
    uint8_t supportsAlternateLayouts;
    int16_t wantLayout;
    int16_t filterCase;
    int16_t dummyPlaneValue;
    void *premiereHook;
    void *advanceState;
    uint8_t supportsAbsolute;
    uint8_t wantsAbsolute;
    void *getPropertyObsolete;
    uint8_t cannotUndo;
    uint8_t supportsPadding;
    int16_t inputPadding;
    int16_t outputPadding;
    int16_t maskPadding;
    uint8_t samplingSupport;
    uint8_t reservedByte;
    int32_t inputRate;
    int32_t maskRate;
    void *colorServices;
    int16_t inLayerPlanes, inTransparencyMask, inLayerMasks, inInvertedLayerMasks, inNonLayerPlanes;
    int16_t outLayerPlanes, outTransparencyMask, outLayerMasks, outInvertedLayerMasks, outNonLayerPlanes;
    int16_t absLayerPlanes, absTransparencyMask, absLayerMasks, absInvertedLayerMasks, absNonLayerPlanes;
    int16_t inPreDummyPlanes, inPostDummyPlanes, outPreDummyPlanes, outPostDummyPlanes;
    int32_t inColumnBytes, inPlaneBytes, outColumnBytes, outPlaneBytes;
    /* 3.0.4 */
    void *imageServicesProcs;
    void *propertyProcs;
    int16_t inTileHeight, inTileWidth;
    Point16 inTileOrigin;
    int16_t absTileHeight, absTileWidth;
    Point16 absTileOrigin;
    int16_t outTileHeight, outTileWidth;
    Point16 outTileOrigin;
    int16_t maskTileHeight, maskTileWidth;
    Point16 maskTileOrigin;
    /* 4.0 */
    void *descriptorParameters;
    void *errorString;
    void *channelPortProcs;
    void *documentInfo;
    /* 5.0 */
    void *sSPBasic;
    void *plugInRef;
    int32_t depth;              /* bits per sample: 8 or 16 */
    /* 6.0 */
    void *iccProfileData;
    int32_t iccProfileSize;
    int32_t canUseICCProfiles;
    uint8_t reserved[54];
} FilterRecord;

#pragma warning(pop)
#pragma pack(pop)

/* Guard the 64-bit layout against accidental edits. The numbers come from the
 * independent ctypes definition in tests/filter_host.py. */
#if defined(_WIN64)
_Static_assert(offsetof(FilterRecord, filterRect) == 34, "FilterRecord layout");
_Static_assert(offsetof(FilterRecord, inRect) == 64, "FilterRecord layout");
_Static_assert(offsetof(FilterRecord, inData) == 88, "FilterRecord layout");
_Static_assert(offsetof(FilterRecord, outData) == 100, "FilterRecord layout");
_Static_assert(offsetof(FilterRecord, imageMode) == 156, "FilterRecord layout");
_Static_assert(offsetof(FilterRecord, depth) == 480, "FilterRecord layout");
_Static_assert(sizeof(FilterRecord) == 556, "FilterRecord layout");
#endif

#endif
