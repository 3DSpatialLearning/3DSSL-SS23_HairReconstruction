#pragma once
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "managed.h"
#include <string.h>        // memset()
#include <vector_types.h>  // float4

class __align__(128) LineState : public Managed {
   public:
    float3 *unitDirection;
    float *depth;
    float *lineCost;

    // TODO: delete
    float4 *norm4;  // 3 values for normal and last for d
    float *c;       // cost
    int n; // What is n? Is this used?
    int s;  // stride
    int l;  // length
    void resize(int n) {
        cudaMallocManaged(&depth, sizeof(float) * n);
        cudaMallocManaged(&unitDirection, sizeof(float3) * n);
        cudaMallocManaged(&lineCost, sizeof(float) * n);
        memset(depth, 0, sizeof(float) * n);
        memset(unitDirection, 0, sizeof(float3) * n);
        memset(lineCost, 0, sizeof(float) * n);

        // TODO: delete
        cudaMallocManaged(&c, sizeof(float) * n);
        cudaMallocManaged(&norm4, sizeof(float4) * n);
        memset(c, 0, sizeof(float) * n);
        memset(norm4, 0, sizeof(float4) * n);
    }
    ~LineState() {
        cudaFree(depth);
        cudaFree(unitDirection);
        cudaFree(lineCost);
        // TODO: delete
        cudaFree(c);
        cudaFree(norm4);
    }
};
