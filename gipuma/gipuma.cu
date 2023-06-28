// #include <helper_math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "config.h"
#include "globalstate.h"
#include "imageinfo.h"
#include "linestate.h"
#include <stdint.h>  // for uint8_t
#include <stdio.h>

#include "helper_cuda.h"
#include "vector_operations.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <vector_types.h>  // float4

using namespace std;

#ifndef SHARED_HARDCODED
__managed__ int SHARED_SIZE_W_m;
__constant__ int SHARED_SIZE_W;
__managed__ int SHARED_SIZE_H;
__managed__ int SHARED_SIZE = 0;
__managed__ int WIN_RADIUS_W;
__managed__ int WIN_RADIUS_H;
__managed__ int TILE_W;
__managed__ int TILE_H;
#endif

__device__ FORCEINLINE_GIPUMA float curand_between(curandState *cs,
                                                   const float &min,
                                                   const float &max) {
    return curand_uniform(cs) * (max - min) + min;
}

__device__ FORCEINLINE_GIPUMA static void rndUnitVectorSphereMarsaglia_cu(
    float3 *v, curandState *cs) {
    float x = 1.0f;
    float y = 1.0f;
    float sum = 2.0f;
    while (sum >= 1.0f) {
        x = curand_between(cs, -1.0f, 1.0f);
        y = curand_between(cs, -1.0f, 1.0f);
        sum = get_pow2_norm(x, y);
    }
    const float sq = sqrtf(1.0f - sum);
    v->x = 2.0f * x * sq;
    v->y = 2.0f * y * sq;
    v->z = 1.0f - 2.0f * sum;
}

__device__ FORCEINLINE_GIPUMA void getPixelWorldCoord_cu(
    const int2 pixel,
    const float depth,
    const float* K_inv,
    const float* Rt_inv,
    float3* res
) {
    float3 pixelHomogenCoord;
    pixelHomogenCoord.x = pixel.x * depth;
    pixelHomogenCoord.y = pixel.y * depth;
    pixelHomogenCoord.z = depth;

    float4 pixelCameraHomogenCoord;
    pixelCameraHomogenCoord.x = K_inv[0] * pixelHomogenCoord.x + K_inv[1] * pixelHomogenCoord.y + K_inv[2] * pixelHomogenCoord.z;
    pixelCameraHomogenCoord.y = K_inv[3] * pixelHomogenCoord.x + K_inv[4] * pixelHomogenCoord.y + K_inv[5] * pixelHomogenCoord.z;
    pixelCameraHomogenCoord.z = K_inv[6] * pixelHomogenCoord.x + K_inv[7] * pixelHomogenCoord.y + K_inv[8] * pixelHomogenCoord.z;
    pixelCameraHomogenCoord.w = 1;

    // Pixel world coordinates
    res->x = Rt_inv[0] * pixelCameraHomogenCoord.x + Rt_inv[1] * pixelCameraHomogenCoord.y + Rt_inv[2] * pixelCameraHomogenCoord.z + Rt_inv[3] * pixelCameraHomogenCoord.w;
    res->y = Rt_inv[4] * pixelCameraHomogenCoord.x + Rt_inv[5] * pixelCameraHomogenCoord.y + Rt_inv[6] * pixelCameraHomogenCoord.z + Rt_inv[7] * pixelCameraHomogenCoord.w;
    res->z = Rt_inv[8] * pixelCameraHomogenCoord.x + Rt_inv[9] * pixelCameraHomogenCoord.y + Rt_inv[10] * pixelCameraHomogenCoord.z + Rt_inv[11] * pixelCameraHomogenCoord.w;

}

__device__ void dummyfloat3(float3* res) {
    res->x = 1.0;
    res->y = 1.0;
    res->z = 1.0;

}
__device__ FORCEINLINE_GIPUMA void getPointPixelCoord_cu(
    float3 point,
    float* P,
    float2* res
) {
    float4 pointHomogenCoord;
    pointHomogenCoord.x = point.x;
    pointHomogenCoord.y = point.y;
    pointHomogenCoord.z = point.z;
    pointHomogenCoord.w = 1;

    float3 pointCameraCoord;
    pointCameraCoord.x = P[0] *  pointHomogenCoord.x + P[1] * pointHomogenCoord.y + P[2] * pointHomogenCoord.z + P[3] * pointHomogenCoord.w;
    pointCameraCoord.y = P[4] *  pointHomogenCoord.x + P[5] * pointHomogenCoord.y + P[6] * pointHomogenCoord.z + P[7] * pointHomogenCoord.w;
    pointCameraCoord.z = P[8] *  pointHomogenCoord.x + P[9] * pointHomogenCoord.y + P[10] * pointHomogenCoord.z + P[11] * pointHomogenCoord.w;

    // Pixel coordinates
    res->x = pointCameraCoord.x / pointCameraCoord.z;
    res->y = pointCameraCoord.y / pointCameraCoord.z;
    
}

__device__ FORCEINLINE_GIPUMA void normalize_cu(float3 *__restrict__ v) {
    const float normSquared = pow2(v->x) + pow2(v->y) + pow2(v->z);
    const float inverse_sqrt = rsqrtf(normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
    v->z *= inverse_sqrt;
}

__device__ FORCEINLINE_GIPUMA void normalize2_cu(float2 *__restrict__ v) {
    const float normSquared = pow2(v->x) + pow2(v->y);
    const float inverse_sqrt = rsqrtf(normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
}


__device__ FORCEINLINE_GIPUMA static void projectSamplePointIn3D_cu(
    float2 samplePoint,
    const float3 pointOnLine,
    const float3 lineUnitDirection,
    const float* K_inv,
    const float* Rt,
    float3* res
) {

    float3 pixelHomogenCoord;
    pixelHomogenCoord.x = samplePoint.x;
    pixelHomogenCoord.y = samplePoint.y;
    pixelHomogenCoord.z = 1;


    float3 pixelCameraCoord;
    pixelCameraCoord.x = K_inv[0] * pixelHomogenCoord.x + K_inv[1] * pixelHomogenCoord.y + K_inv[2] * pixelHomogenCoord.z;
    pixelCameraCoord.y = K_inv[3] * pixelHomogenCoord.x + K_inv[4] * pixelHomogenCoord.y + K_inv[5] * pixelHomogenCoord.z;
    pixelCameraCoord.z = K_inv[6] * pixelHomogenCoord.x + K_inv[7] * pixelHomogenCoord.y + K_inv[8] * pixelHomogenCoord.z;


    const float t1 = Rt[3];
    const float t2 = Rt[7];
    const float t3 = Rt[11];


    const float xPrim = pixelCameraCoord.x;
    const float yPrim = pixelCameraCoord.y;

    float3 r1;
    r1.x = Rt[0];
    r1.y = Rt[1];
    r1.z = Rt[2];

    float3 r2;
    r2.x = Rt[4];
    r2.y = Rt[5];
    r2.z = Rt[6];

    float3 r3;
    r3.x = Rt[8];
    r3.y = Rt[9];
    r3.z = Rt[10];

    float lambda = 0;
    // For some reason brackets are VERY important here!
    const float d = dot4(r2, lineUnitDirection) - (yPrim * (dot4(r3, lineUnitDirection)));
    const float n = (yPrim * (dot4(r3, pointOnLine))) + (yPrim * t3) - (dot4(r2, pointOnLine)) - (t2);
    const float d2 = dot4(r1, lineUnitDirection) - (xPrim * (dot4(r3, lineUnitDirection)));
    const float n2 = (xPrim * (dot4(r3, pointOnLine))) + (xPrim * t3) - (dot4(r1, pointOnLine)) - (t1);
    
    if (abs(d) < 0.001f) {
        lambda = n2 / d2;
    } else {
        lambda = n / d;
    }

    res->x = lineUnitDirection.x * lambda + pointOnLine.x;
    res->y = lineUnitDirection.y * lambda + pointOnLine.y;
    res->z = lineUnitDirection.z * lambda + pointOnLine.z;
}


template <typename T>
__device__ FORCEINLINE_GIPUMA void samplePoints_cu(
    const GlobalState &gs,
    const int2 pixelCoord,
    const float depth,
    const float3 unitDirection,
    float2 samples[400]
) {

    const int k = gs.params->k;
    const int rk = gs.params->rk;
    const int selectedViewsNumber = gs.cameras->viewSelectionSubsetNumber;
    int* selectedViewsSubset = gs.cameras->viewSelectionSubset;
    const int referenceImageIndex = 0;
    const int u = pixelCoord.x;
    const int v = pixelCoord.y;

    // correct
    float3 pixelWorldCoord;
    getPixelWorldCoord_cu(
        pixelCoord,
        depth,
        gs.cameras->cameras[referenceImageIndex].K_inv,
        gs.cameras->cameras[referenceImageIndex].Rt_extended_inv,
        &pixelWorldCoord
    );

    // correct
    float3 seoncdPointOnLineWorldCoord;
    addout(pixelWorldCoord, unitDirection, seoncdPointOnLineWorldCoord);
    
    // correct
    float2 seoncdPointPixelCoord;
    getPointPixelCoord_cu(seoncdPointOnLineWorldCoord, gs.cameras->cameras[referenceImageIndex].P, &seoncdPointPixelCoord);

    
    // correct
    float2 lineInReferenceImageUnitDirection;
    subout2(pixelCoord, seoncdPointPixelCoord, lineInReferenceImageUnitDirection);
    
    
    // correct
    normalize2_cu(&lineInReferenceImageUnitDirection);

    
    // correct
    float2 samplesInReferenceView[50];
    int sampleIdx = 0;

    for (int i = -k / 2; i <= k / 2; i++) {
        float scale = 2 * i * rk / k;

        samplesInReferenceView[sampleIdx].x = pixelCoord.x + scale * lineInReferenceImageUnitDirection.x;
        samplesInReferenceView[sampleIdx].y = pixelCoord.y + scale * lineInReferenceImageUnitDirection.y;

        sampleIdx += 1;
    }

    for (int j = 0; j < k; j++) {
        float2 samplePixel = samplesInReferenceView[j];
        // correct
        float3 sampleWorldCoord;
        projectSamplePointIn3D_cu(samplePixel, pixelWorldCoord, unitDirection, gs.cameras->cameras[referenceImageIndex].K_inv, gs.cameras->cameras[referenceImageIndex].Rt, &sampleWorldCoord);
        
        for (int i = 0; i < selectedViewsNumber; i++) {
            int cameraIdx = selectedViewsSubset[i];

            // correct
            float2 samplePixelInImageI;
            getPointPixelCoord_cu(sampleWorldCoord, gs.cameras->cameras[cameraIdx].P, &samplePixelInImageI);
            samples[i * k + j] = samplePixelInImageI;
        }
    }

}



template <typename T>
__device__ FORCEINLINE_GIPUMA static float pmCostMultiview_cu(
    const GlobalState &gs,
    const int2 pixelCoord,
    const float depth,
    const float3 unitDirection
) {
        float cost = 0.f;
        // TODO: remove check
        if(pixelCoord.x != 196 || pixelCoord.y != 1225) {
            return cost;
        }
        
        const int k = gs.params->k;
        float2 samples[400];

        samplePoints_cu<T>(gs, pixelCoord, depth, unitDirection, samples);

        for (int j = 0; j < 1; j++){
            printf("after j: %d x:%f y:%f\n", j, samples[j].x, samples[j].y);
        }
        // TODO: compute geometric cost
        float geometricCost = 0.f;

        // TODO: compute intensity cost
        float intensityCost = 0.f; 
        const int rk = gs.params->rk;
        const int selectedViewsNumber = gs.cameras->viewSelectionSubsetNumber;
        int* selectedViewsSubset = gs.cameras->viewSelectionSubset;
        const int referenceImageIndex = 0;
        const int rows = gs.cameras->rows;
        const int cols = gs.cameras->cols;

        const cudaTextureObject_t referenceImg = gs.imgs[referenceImageIndex];

        for (int i = 1; i < selectedViewsNumber; i++) {
            int cameraIdx = selectedViewsSubset[i];
            const cudaTextureObject_t otherImg = gs.imgs[cameraIdx];
            float numerator = 0.f;
            float denominator1 = 0;
            float denominator2 = 0;
            float referenceImageIntensitySum = 0.f;
            float otherImageIntensitySum = 0.f;
            float validSamples = 0.f;

            for (int j = 0; j < k; j++){
                if (samples[j].x < 0 || samples[j].y < 0 || samples[j].x >= cols || samples[j].y >= rows) {
                    continue;
                }

                if (samples[i * k + j].x < 0 || samples[i * k + j].y < 0 ||
                    samples[i * k + j].x >= cols || samples[i * k + j].y >= rows
                ) {
                    continue;
                }
                validSamples += 1;
                referenceImageIntensitySum += texat(referenceImg, samples[j].x, samples[j].y);
                otherImageIntensitySum += texat(otherImg, samples[i * k + j].x, samples[i * k + j].y);
            }

            if(validSamples == 0 ){
                // What to do here?
                intensityCost += 1000.f;
                printf("WARNING! validSamples is 0\n");
                continue;
            }
            const float referenceImageIntensityMean = referenceImageIntensitySum / validSamples;
            const float otherImageIntensityMean = otherImageIntensitySum / validSamples;

            for (int j = 0; j < k; j++){
                if (samples[j].x < 0 || samples[j].y < 0 || samples[j].x >= cols || samples[j].y >= rows) {
                    continue;
                }

                if (samples[i * k + j].x < 0 || samples[i * k + j].y < 0 ||
                    samples[i * k + j].x >= cols || samples[i * k + j].y >= rows
                ) {
                    continue;
                }
                float intensityReference = texat(referenceImg, samples[j].x, samples[j].y) - referenceImageIntensityMean;
                float intensityOther = texat(otherImg, samples[i * k + j].x, samples[i * k + j].y) - otherImageIntensityMean;

                numerator += intensityReference * intensityOther;
                denominator1 += intensityOther * intensityOther;
                denominator2 += intensityReference * intensityReference;
            }
            float denominator = denominator1 * denominator2;
            if(denominator == 0) {
                // What to do here?
                intensityCost += 1000.f;
                printf("WARNING! denominator is 0\n");
            } else {
                float correlation = numerator / denominator;
                intensityCost += correlation;
            }
        }
        float alpha = 0.1;
        cost = (1 - alpha) * geometricCost + alpha * intensityCost;
        return cost;
}

template <typename T>
__global__ void gipuma_init_cu2(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    const int center = p.y * cols + p.x;
    curandState localState = gs.cs[center];
    curand_init(clock64(), p.y, p.x, &localState);

    float3 generatedUnitDir;
    rndUnitVectorSphereMarsaglia_cu(&generatedUnitDir, &localState);
    gs.lines->unitDirection[center] = generatedUnitDir;
    // gs.lines->unitDirection[center].x = -0.12800153;
    // gs.lines->unitDirection[center].y = -0.68858582;
    // gs.lines->unitDirection[center].z = 0.7137683;

    // use disparity instead of depth?
    float mind = gs.params->depthMin;
    float maxd = gs.params->depthMax;
    gs.lines->depth[center] = curand_between(&localState, mind, maxd);
    // gs.lines->depth[center] = 1.0675795;

    gs.lines->lineCost[center] =  pmCostMultiview_cu<T>(
        gs,
        p,
        gs.lines->depth[center],
        gs.lines->unitDirection[center]
    );
    return;
}

template <typename T>
__global__ void gipuma_black_spatialPropClose_cu(GlobalState &gs) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;
    printf("gipuma_black_spatialPropClose_cu x, y: %d %d \n", p.x, p.y);
    int2 tile_offset;
    tile_offset.x = blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    // gipuma_checkerboard_spatialPropClose_cu<T>(gs, p, tile_offset, iter);
}

template <typename T>
__global__ void gipuma_black_spatialPropFar_cu(GlobalState &gs) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;
    int2 tile_offset;
    tile_offset.x = blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    printf("gipuma_black_spatialPropFar_cu x, y: %d %d \n", p.x, p.y);
    // gipuma_checkerboard_spatialPropFar_cu<T>(gs, p, tile_offset, iter);
}

template <typename T>
__global__ void gipuma_black_lineRefine_cu(GlobalState &gs) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;
    int2 tile_offset;
    tile_offset.x = blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    printf("gipuma_black_lineRefine_cu x, y: %d %d \n", p.x, p.y);

    // gipuma_checkerboard_lineRefinement_cu<T>(gs, p, tile_offset, iter);
}

template <typename T>
__global__ void gipuma_red_spatialPropClose_cu(GlobalState &gs) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;
    int2 tile_offset;
    tile_offset.x = blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    printf("gipuma_red_spatialPropClose_cu x, y: %d %d \n", p.x, p.y);

    // gipuma_checkerboard_spatialPropClose_cu<T>(gs, p, tile_offset, iter);
}

template <typename T>
__global__ void gipuma_red_spatialPropFar_cu(GlobalState &gs) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;
    int2 tile_offset;
    tile_offset.x = blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    printf("gipuma_red_spatialPropFar_cu x, y: %d %d \n", p.x, p.y);

    // gipuma_checkerboard_spatialPropFar_cu<T>(gs, p, tile_offset, iter);
}

template <typename T>
__global__ void gipuma_red_lineRefine_cu(GlobalState &gs) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;
    int2 tile_offset;
    tile_offset.x = blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    printf("gipuma_red_lineRefine_cu x, y: %d %d \n", p.x, p.y);

    // gipuma_checkerboard_lineRefinement_cu<T>(gs, p, tile_offset, iter);
}

float printTotalCost(GlobalState gs) {
    float c = 0;
    printf("gs.cameras->rows: %d\n", gs.cameras->rows);
    printf("gs.cameras->cols: %d\n", gs.cameras->cols);
    for (int i = 0; i < gs.cameras->rows; i++) {
        for (int j = 0; j < gs.cameras->cols; j++) {
            c += gs.lines->lineCost[i * gs.cameras->cols + j];
        }
    }

    printf("c: %f\n\n\n\n", c);
    return c;
}

template <typename T>
void gipuma(GlobalState &gs) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&gs.cs, rows * cols * sizeof(curandState)));

    // int SHARED_SIZE_W_host;
#ifndef SHARED_HARDCODED
    int blocksize_w =
        gs.params->box_hsize + 1;  // +1 for the gradient computation
    int blocksize_h =
        gs.params->box_vsize + 1;  // +1 for the gradient computation
    WIN_RADIUS_W = (blocksize_w) / (2);
    WIN_RADIUS_H = (blocksize_h) / (2);

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);
    TILE_W = BLOCK_W;
    TILE_H = BLOCK_H * 2;
    SHARED_SIZE_W_m = (TILE_W + WIN_RADIUS_W * 2);
    SHARED_SIZE_H = (TILE_H + WIN_RADIUS_H * 2);
    SHARED_SIZE = (SHARED_SIZE_W_m * SHARED_SIZE_H);
    cudaMemcpyToSymbol(SHARED_SIZE_W, &SHARED_SIZE_W_m,
                       sizeof(SHARED_SIZE_W_m));
    // SHARED_SIZE_W_host = SHARED_SIZE_W_m;
#else
    // SHARED_SIZE_W_host = SHARED_SIZE;
#endif
    int shared_size_host = SHARED_SIZE;

    dim3 grid_size;
    grid_size.x = (cols + BLOCK_W - 1) / BLOCK_W;
    grid_size.y = ((rows / 2) + BLOCK_H - 1) / BLOCK_H;
    dim3 block_size;
    block_size.x = BLOCK_W;
    block_size.y = BLOCK_H;

    dim3 grid_size_initrand;
    grid_size_initrand.x = (cols + 16 - 1) / 16;
    grid_size_initrand.y = (rows + 16 - 1) / 16;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    int maxiter = gs.params->iterations;
    printf("Device memory used: %fMB\n", used / 1000000.0f);
    printf("Blocksize is %dx%d\n", gs.params->box_hsize, gs.params->box_vsize);

    printf("Number of iterations is %d\n", maxiter);
    gipuma_init_cu2<T><<<grid_size_initrand, block_size_initrand>>>(gs);

    cudaDeviceSynchronize();

    printTotalCost(gs);
    cudaEventRecord(start);
    // for (int it =0;it<gs.params.iterations; it++) {
    printf("Iteration ");
    for (int it = 0; it < 0; it++) {
        // for (int it = 0; it < maxiter; it++) {
        printf("%d ", it + 1);
        // spatial propagation of 4 closest neighbors (1px up/down/left/right)
        gipuma_black_spatialPropClose_cu<T>
            <<<grid_size, block_size, shared_size_host * sizeof(T)>>>(gs);
        cudaDeviceSynchronize();

        // spatial propagation of 4 far away neighbors (5px up/down/left/right)
        gipuma_black_spatialPropFar_cu<T>
            <<<grid_size, block_size, shared_size_host * sizeof(T)>>>(gs);
        cudaDeviceSynchronize();

        // plane refinement
        gipuma_black_lineRefine_cu<T>
            <<<grid_size, block_size, shared_size_host * sizeof(T)>>>(gs);
        cudaDeviceSynchronize();

        // spatial propagation of 4 closest neighbors (1px up/down/left/right)
        gipuma_red_spatialPropClose_cu<T>
            <<<grid_size, block_size, shared_size_host * sizeof(T)>>>(gs);
        cudaDeviceSynchronize();

        // spatial propagation of 4 far away neighbors (5px up/down/left/right)
        gipuma_red_spatialPropFar_cu<T>
            <<<grid_size, block_size, shared_size_host * sizeof(T)>>>(gs);
        cudaDeviceSynchronize();

        // plane refinement
        gipuma_red_lineRefine_cu<T>
            <<<grid_size, block_size, shared_size_host * sizeof(T)>>>(gs);
        cudaDeviceSynchronize();
    }
    printf("\n");
    printf("here?\n");
    // printf("Computing final disparity\n");
    // gipuma_compute_disp<<<grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\t\tTotal time needed for computation: %f seconds\n",
           milliseconds / 1000.f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // print results to file
    cudaFree(&gs.cs);
}

int runcuda(GlobalState &gs) {
    gipuma<float>(gs);
    return 0;
}
