#pragma once
#include <cstdint>

// Private Pillow 12.3 raster primitives, adapted to C++. RGB/RGBA canvases use
// four bytes per pixel; font masks are composited by the C++ text renderer.
namespace ezlocalai::pillow {
using UINT8 = uint8_t;
using UINT16 = uint16_t;
using INT32 = int32_t;
struct ImagingMemoryInstance {
    int xsize = 0, ysize = 0, mode = 0;
    UINT8 **image8 = nullptr;
    INT32 **image32 = nullptr;
    char **image = nullptr;
};
using Imaging = ImagingMemoryInstance *;
int ImagingDrawPoint(Imaging, int, int, const void *, int);
int ImagingDrawLine(Imaging, int, int, int, int, const void *, int);
int ImagingDrawWideLine(Imaging, int, int, int, int, const void *, int, int, Imaging);
int ImagingDrawRectangle(Imaging, int, int, int, int, const void *, int, int, int);
int ImagingDrawPolygon(Imaging, int, int *, const void *, int, int, int, Imaging);
int ImagingDrawEllipse(Imaging, int, int, int, int, const void *, int, int, int);
int ImagingDrawArc(Imaging, int, int, int, int, float, float, const void *, int, int);
int ImagingDrawPieslice(Imaging, int, int, int, int, float, float, const void *, int, int, int);
}
