#pragma once

#include "helper_cuda.h"
#include "managed.h"
#include <cuda_runtime.h>
#include <vector_types.h>

class Line_cu : public Managed {
   public:
    // Z coord of 3D point in reference camera frame coord system
    float d;
    // Normalized direction of line in world space
    // Note: world space matched reference camera sapce
    float4 unitDirection;

    Line_cu() { d = 0.f; }

    ~Line_cu() {}
};
