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

template <typename T>
__global__ void gipuma_init_cu2(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;
    if (p.x >= cols) return;
    if (p.y >= rows) return;
    // printf("gipuma_init_cu2 x, y: %d %d \n", p.x, p.y);

    const int center = p.y * cols + p.x;
    curandState localState = gs.cs[center];
    curand_init(clock64(), p.y, p.x, &localState);

    float3 generatedUnitDir;
    rndUnitVectorSphereMarsaglia_cu(&generatedUnitDir, &localState);
    gs.lines->unitDirection[center] = generatedUnitDir;

    // use disparity instead of depth?
    float mind = gs.params->depthMin;
    float maxd = gs.params->depthMax;
    gs.lines->depth[center] = curand_between(&localState, mind, maxd);

    // TODO: compute and save cost
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
