#pragma once

#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "imageinfo.h"
#include "managed.h"
#include "lineMap_cu.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

class GlobalState : public Managed {
public:
    CameraParameters_cu *cameras;
    LineState *lines;
    LineMap_cu *lineMap;
    curandState *cs;
    AlgorithmParameters *params;

    cudaTextureObject_t imgs  [MAX_IMAGES];
    cudaArray *cuArray[MAX_IMAGES];
    //cudaTextureObject_t gradx [MAX_IMAGES];
    //cudaTextureObject_t grady [MAX_IMAGES];
    GlobalState() {
        //printf("GlobalState constructor\n");
        cameras = new CameraParameters_cu;
        lines = new LineState;
        lineMap = new LineMap_cu;
    }
    ~GlobalState() {
        //printf("GlobalState destructor\n");
        delete cameras;
        delete lines;
        delete lineMap;
    }
};
