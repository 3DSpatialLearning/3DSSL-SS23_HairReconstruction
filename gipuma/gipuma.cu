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

__device__ FORCEINLINE_GIPUMA void getPixelWorldCoord_cu(const int2 pixel,
                                                         const float depth,
                                                         const float *K_inv,
                                                         const float *Rt_inv,
                                                         float3 *res) {
    float3 pixelHomogenCoord;
    pixelHomogenCoord.x = pixel.x * depth;
    pixelHomogenCoord.y = pixel.y * depth;
    pixelHomogenCoord.z = depth;

    float4 pixelCameraHomogenCoord;
    pixelCameraHomogenCoord.x = K_inv[0] * pixelHomogenCoord.x +
                                K_inv[1] * pixelHomogenCoord.y +
                                K_inv[2] * pixelHomogenCoord.z;
    pixelCameraHomogenCoord.y = K_inv[3] * pixelHomogenCoord.x +
                                K_inv[4] * pixelHomogenCoord.y +
                                K_inv[5] * pixelHomogenCoord.z;
    pixelCameraHomogenCoord.z = K_inv[6] * pixelHomogenCoord.x +
                                K_inv[7] * pixelHomogenCoord.y +
                                K_inv[8] * pixelHomogenCoord.z;
    pixelCameraHomogenCoord.w = 1;

    // Pixel world coordinates
    res->x = Rt_inv[0] * pixelCameraHomogenCoord.x +
             Rt_inv[1] * pixelCameraHomogenCoord.y +
             Rt_inv[2] * pixelCameraHomogenCoord.z +
             Rt_inv[3] * pixelCameraHomogenCoord.w;
    res->y = Rt_inv[4] * pixelCameraHomogenCoord.x +
             Rt_inv[5] * pixelCameraHomogenCoord.y +
             Rt_inv[6] * pixelCameraHomogenCoord.z +
             Rt_inv[7] * pixelCameraHomogenCoord.w;
    res->z = Rt_inv[8] * pixelCameraHomogenCoord.x +
             Rt_inv[9] * pixelCameraHomogenCoord.y +
             Rt_inv[10] * pixelCameraHomogenCoord.z +
             Rt_inv[11] * pixelCameraHomogenCoord.w;
}

__device__ FORCEINLINE_GIPUMA void getPointPixelCoord_cu(float3 point, float *P,
                                                         float2 *res) {
    float4 pointHomogenCoord;
    pointHomogenCoord.x = point.x;
    pointHomogenCoord.y = point.y;
    pointHomogenCoord.z = point.z;
    pointHomogenCoord.w = 1;

    float3 pointCameraCoord;
    pointCameraCoord.x =
        P[0] * pointHomogenCoord.x + P[1] * pointHomogenCoord.y +
        P[2] * pointHomogenCoord.z + P[3] * pointHomogenCoord.w;
    pointCameraCoord.y =
        P[4] * pointHomogenCoord.x + P[5] * pointHomogenCoord.y +
        P[6] * pointHomogenCoord.z + P[7] * pointHomogenCoord.w;
    pointCameraCoord.z =
        P[8] * pointHomogenCoord.x + P[9] * pointHomogenCoord.y +
        P[10] * pointHomogenCoord.z + P[11] * pointHomogenCoord.w;

    // Pixel coordinates
    res->x = pointCameraCoord.x / pointCameraCoord.z;
    res->y = pointCameraCoord.y / pointCameraCoord.z;
}

__device__ FORCEINLINE_GIPUMA void normalize_cu(float2 *__restrict__ v) {
    const float normSquared = pow2(v->x) + pow2(v->y);
    const float inverse_sqrt = rsqrtf(normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
}

__device__ FORCEINLINE_GIPUMA static void projectSamplePointIn3D_cu(
    float2 samplePoint, const float3 pointOnLine,
    const float3 lineUnitDirection, const float *K_inv, const float *Rt,
    float3 *res) {
    float3 pixelHomogenCoord;
    pixelHomogenCoord.x = samplePoint.x;
    pixelHomogenCoord.y = samplePoint.y;
    pixelHomogenCoord.z = 1;

    float3 pixelCameraCoord;
    pixelCameraCoord.x = K_inv[0] * pixelHomogenCoord.x +
                         K_inv[1] * pixelHomogenCoord.y +
                         K_inv[2] * pixelHomogenCoord.z;
    pixelCameraCoord.y = K_inv[3] * pixelHomogenCoord.x +
                         K_inv[4] * pixelHomogenCoord.y +
                         K_inv[5] * pixelHomogenCoord.z;
    pixelCameraCoord.z = K_inv[6] * pixelHomogenCoord.x +
                         K_inv[7] * pixelHomogenCoord.y +
                         K_inv[8] * pixelHomogenCoord.z;

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
    const float d =
        dot4(r2, lineUnitDirection) - (yPrim * (dot4(r3, lineUnitDirection)));
    const float n = (yPrim * (dot4(r3, pointOnLine))) + (yPrim * t3) -
                    (dot4(r2, pointOnLine)) - (t2);
    const float d2 =
        dot4(r1, lineUnitDirection) - (xPrim * (dot4(r3, lineUnitDirection)));
    const float n2 = (xPrim * (dot4(r3, pointOnLine))) + (xPrim * t3) -
                     (dot4(r1, pointOnLine)) - (t1);

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
__device__ FORCEINLINE_GIPUMA void samplePoints_cu(const GlobalState &gs,
                                                   const int2 pixelCoord,
                                                   const float depth,
                                                   const float3 unitDirection,
                                                   float2 samples[400]) {
    const int k = gs.params->k;
    const int rk = gs.params->rk;
    const int selectedViewsNumber = gs.cameras->viewSelectionSubsetNumber;
    int *selectedViewsSubset = gs.cameras->viewSelectionSubset;

    // correct
    float3 pixelWorldCoord;
    getPixelWorldCoord_cu(
        pixelCoord, depth, gs.cameras->cameras[REFERENCE].K_inv,
        gs.cameras->cameras[REFERENCE].Rt_extended_inv,
        &pixelWorldCoord);

    // printf("pixelWorldCoord %f %f %f \n", pixelWorldCoord.x, pixelWorldCoord.y, pixelWorldCoord.z);
    // correct
    float3 seoncdPointOnLineWorldCoord;

    addout(pixelWorldCoord, unitDirection, seoncdPointOnLineWorldCoord);
    // printf("seoncdPointOnLineWorldCoord %f %f %f \n", seoncdPointOnLineWorldCoord.x, seoncdPointOnLineWorldCoord.y, seoncdPointOnLineWorldCoord.z);

    // correct
    float2 seoncdPointPixelCoord;
    getPointPixelCoord_cu(seoncdPointOnLineWorldCoord,
                          gs.cameras->cameras[REFERENCE].P,
                          &seoncdPointPixelCoord);
    // printf("seoncdPointPixelCoord %f %f \n", seoncdPointPixelCoord.x, seoncdPointPixelCoord.y);

    // correct
    float2 lineInReferenceImageUnitDirection;
    subout2(pixelCoord, seoncdPointPixelCoord,
            lineInReferenceImageUnitDirection);
    // printf("lineInReferenceImageUnitDirection %f %f \n", lineInReferenceImageUnitDirection.x, lineInReferenceImageUnitDirection.y);

    // correct
    normalize_cu(&lineInReferenceImageUnitDirection);
    // printf("normalized lineInReferenceImageUnitDirection %f %f \n", lineInReferenceImageUnitDirection.x, lineInReferenceImageUnitDirection.y);

    // correct
    int sampleIdx = 0;

    for (int i = -k / 2; i <= k / 2; i++) {

       float scale = 2 * (float)i * (float)rk / (float)k;

        samples[sampleIdx].x =
            pixelCoord.x + scale * lineInReferenceImageUnitDirection.x;
        samples[sampleIdx].y =
            pixelCoord.y + scale * lineInReferenceImageUnitDirection.y;

        sampleIdx += 1;
    }

    for (int j = 0; j < k; j++) {
        float2 samplePixel = samples[j];
        // correct
        float3 sampleWorldCoord;
        projectSamplePointIn3D_cu(
            samplePixel, pixelWorldCoord, unitDirection,
            gs.cameras->cameras[REFERENCE].K_inv,
            gs.cameras->cameras[REFERENCE].Rt, &sampleWorldCoord);

        for (int i = 0; i < selectedViewsNumber; i++) {
            int cameraIdx = selectedViewsSubset[i];

            // correct
            float2 samplePixelInImageI;
            getPointPixelCoord_cu(sampleWorldCoord,
                                  gs.cameras->cameras[cameraIdx].P,
                                  &samplePixelInImageI);
            samples[cameraIdx * k + j] = samplePixelInImageI;
        }
    }
}

template <typename T>
__device__ FORCEINLINE_GIPUMA static float angularDiff_cu(
    const float2 lineDirection, const float orientationAngle) {
    float lineDirectionX = lineDirection.x;
    float lineDirectionY = lineDirection.y;

    if (lineDirectionY < 0) {
        lineDirectionY = lineDirectionY * (-1);
        lineDirectionX = lineDirectionX * (-1);
    }

    float lineAngle = atan2(lineDirectionY, lineDirectionX);
    float angleDiff = abs(lineAngle - orientationAngle);

    return angleDiff;
}

template <typename T>
__device__ FORCEINLINE_GIPUMA static int validNeighborsCount_cu(
    const GlobalState &gs, const float2 samples[400]) {
    const int k = gs.params->k;
    const int selectedViewsNumber = gs.cameras->viewSelectionSubsetNumber;
    int *selectedViewsSubset = gs.cameras->viewSelectionSubset;

    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    int validNeighborsCount = 0;
    for (int i = 0; i < selectedViewsNumber; i++) {
        int validSamplesCount = 0;
        const int cameraIdx = selectedViewsSubset[i];

        for (int j = 0; j < k; j++) {
            if (samples[j].x < 0 || samples[j].y < 0 || samples[j].x >= cols ||
                samples[j].y >= rows) {
                continue;
            }
            const float2 sampleInOtherView = samples[cameraIdx * k + j];
            if (sampleInOtherView.x < 0 || sampleInOtherView.y < 0 ||
                sampleInOtherView.x >= cols || sampleInOtherView.y >= rows) {
                continue;
            }
            validSamplesCount++;
        }

        if (validSamplesCount == 0) {
            // printf("WARNING valid sample count is 0!\n");
            continue;
        }
        validNeighborsCount++;
    }

    return validNeighborsCount;
}

template <typename T>
__device__ FORCEINLINE_GIPUMA static float geometricCost_cu(
    const GlobalState &gs, const float2 samples[400], const int cameraIdx) {
    const int k = gs.params->k;

    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    const cudaTextureObject_t imgOrient = gs.orientaionMap[cameraIdx];
    const cudaTextureObject_t imgConf = gs.confidenceValue[cameraIdx];

    float geometricCost = 0.f;
    float confidenceValuesSum = 0.f;

    float2 lineDirection;
    lineDirection.x = samples[cameraIdx * k + k - 1].x - samples[cameraIdx * k].x;
    lineDirection.y = samples[cameraIdx * k + k - 1].y - samples[cameraIdx * k].y;
    int validSamplesCount = 0;
    for (int j = 0; j < k; j++) {
        if (samples[j].x < 0 || samples[j].y < 0 || samples[j].x >= cols ||
            samples[j].y >= rows) {
            continue;
        }

        const float2 sampleInOtherView = samples[cameraIdx * k + j];
        if (sampleInOtherView.x < 0 || sampleInOtherView.y < 0 ||
            sampleInOtherView.x >= cols || sampleInOtherView.y >= rows) {
            continue;
        }

        validSamplesCount++;
        const float orientaion = texat(imgOrient, sampleInOtherView.x, sampleInOtherView.y);
        const float confidence = texat(imgConf, sampleInOtherView.x, sampleInOtherView.y);

        confidenceValuesSum += confidence;

        geometricCost += confidence * angularDiff_cu<T>(lineDirection, orientaion);
    }

    if (confidenceValuesSum == 0) {
        // printf("WARNING confidenceValuesSum is 0! cameraIdx %d\n", cameraIdx);
        return 1000.f;
    }

    geometricCost /= confidenceValuesSum;
    return geometricCost;
}

template <typename T>
__device__ FORCEINLINE_GIPUMA static float intensityCost_cu(
    const GlobalState &gs, const float2 samples[400], const int cameraIdx) {
    const int k = gs.params->k;
    float intensityCost = 0.f;

    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    const cudaTextureObject_t referenceImg = gs.imgs[REFERENCE];
    const cudaTextureObject_t otherImg = gs.imgs[cameraIdx];
    float referenceImageIntensitySum = 0.f;
    float otherImageIntensitySum = 0.f;
    int validSamples = 0;

    for (int j = 0; j < k; j++) {
        if (samples[j].x < 0 || samples[j].y < 0 || samples[j].x >= cols ||
            samples[j].y >= rows) {
            continue;
        }
        float2 sampleInOtherImg = samples[cameraIdx * k + j];
        if (sampleInOtherImg.x < 0 || sampleInOtherImg.y < 0 ||
            sampleInOtherImg.x >= cols || sampleInOtherImg.y >= rows) {
            continue;
        }
        validSamples += 1;
        referenceImageIntensitySum += (texat(referenceImg, samples[j].x, samples[j].y));
        otherImageIntensitySum += (texat(otherImg, sampleInOtherImg.x, sampleInOtherImg.y));
    }

    if (validSamples == 0) {
        // What to do here?
        intensityCost = MAXCOST;
        // printf("WARNING! validSamples is 0\n");
        return intensityCost;
    }

    const float referenceImageIntensityMean =
        referenceImageIntensitySum / validSamples;
    const float otherImageIntensityMean =
        otherImageIntensitySum / validSamples;

    float numerator = 0.f;
    float denominator1 = 0.f;
    float denominator2 = 0.f;
    for (int j = 0; j < k; j++) {
        if (samples[j].x < 0 || samples[j].y < 0 || samples[j].x >= cols ||
            samples[j].y >= rows) {
            continue;
        }

        float2 sampleInOtherImg = samples[cameraIdx * k + j];
        if (sampleInOtherImg.x < 0 || sampleInOtherImg.y < 0 ||
            sampleInOtherImg.x >= cols || sampleInOtherImg.y >= rows) {
            continue;
        }
        float intensityReference = texat(referenceImg, samples[j].x, samples[j].y) - referenceImageIntensityMean;
        float intensityOther = texat(otherImg, sampleInOtherImg.x, sampleInOtherImg.y) - otherImageIntensityMean;

        numerator += (intensityReference * intensityOther);
        denominator1 += (intensityOther * intensityOther);
        denominator2 += (intensityReference * intensityReference);
    }
    float denominator = sqrtf(denominator1 * denominator2);
    float corelation;
    if (denominator < 0.001) {
        // What to do here?
        // printf("WARNING! denominator is 0\n");
        corelation = 1.f;
    }  else {
        corelation = numerator / denominator;

    }

    intensityCost = (1 - corelation) / (float)validSamples;

    return intensityCost;
}

static __device__ FORCEINLINE_GIPUMA void sort_small(float *__restrict__ d,
                                                     const int n) {
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--) d[j] = d[j - 1];
        d[j] = tmp;
    }
}


template <typename T>
__device__ FORCEINLINE_GIPUMA static float getCombinedCosts_cu(
    const float geometricCostRefImage,
    float geometricCosts[20],
    const float intensityCostRefImage,
    float intensityCosts[20],
    int selectedViewsNumber,
    int cost_comb,
    int n_best,
    const int validNeighborsCount
) {

    if (validNeighborsCount == 0) {
        printf("getCombinedCosts_cu WARNING valid view count is 0!\n");
        printf("getCombinedCosts_cu WARNING valid view count is 0!\n");
        printf("getCombinedCosts_cu WARNING valid view count is 0!\n");
        return MAXCOST;
    }
    // printf("geometricCostRefImage %f \n", geometricCostRefImage);
    sort_small(geometricCosts, selectedViewsNumber);

    int numBest = validNeighborsCount; 
    // int numBest = validViewsCount; 
    if (cost_comb == COMB_BEST_N)
        numBest = min(numBest, n_best);

    float totalGeometricCost = 0.f;
    totalGeometricCost += numBest * geometricCostRefImage;
    for (int i = 0; i < numBest; i++) {
        totalGeometricCost += geometricCosts[i];
    }
    totalGeometricCost = totalGeometricCost / (2*(float)numBest + 1);
    // totalGeometricCost = totalGeometricCost / (2*(float)numBest);

    if(totalGeometricCost < -0.01f) {
        printf("geometric cost is less than 0! %f", totalGeometricCost);
    }


    sort_small(intensityCosts, selectedViewsNumber);

    float totalIntensityCost = 0.f;
    totalIntensityCost += intensityCostRefImage;
    for (int i = 0; i < numBest; i++) {
        totalIntensityCost += intensityCosts[i];
    }
    totalIntensityCost = totalIntensityCost / ((float)numBest + 1);
    // totalIntensityCost = totalIntensityCost / ((float)numBest);


    if(totalIntensityCost < -0.01f) {
        printf("inten cost is less than 0! %f", totalIntensityCost);
    }
    float alpha = 0.1;
    float cost = (1 - alpha) * totalGeometricCost + alpha * totalIntensityCost;

    if (cost == 0) {
        printf("COST IS 0, validNeighborsCount %d, totalGeometricCost %f, totalIntensityCost %f numBest %d, geometricCosts[0] 1, 2, 3: %f %f %f %f\n", validNeighborsCount,totalGeometricCost, totalIntensityCost, numBest,
        geometricCosts[0],
        geometricCosts[1],
        geometricCosts[2],
        geometricCosts[3]
        );
    }
    return cost;
}

template <typename T>
__device__ FORCEINLINE_GIPUMA static float pmCostMultiview_cu(
    const GlobalState &gs, const int2 pixelCoord, const float depth,
    const float3 unitDirection) {

    const int selectedViewsNumber = gs.cameras->viewSelectionSubsetNumber;
    int *selectedViewsSubset = gs.cameras->viewSelectionSubset;
    // if(pixelCoord.x != 196 || pixelCoord.y  != 1225) {
    //     return 0.f;
    // }


    float2 samples[400];
    samplePoints_cu<T>(gs, pixelCoord, depth, unitDirection, samples);

    // for (int i =0 ; i< gs.params->k; i++) {
    //     printf("cuda %d %f %f \n", i, samples[i].x, samples[i].y);
    // }


    const int validNeighborsCount = validNeighborsCount_cu<T>(gs, samples);


    if (validNeighborsCount == 0) {
        // printf("WARNING valid view count is 0!\n");
        return MAXCOST;
    }

    float geometricCosts[20];
    const float geometricCostRefImage = geometricCost_cu<T>(gs, samples, 0);
    for (int i = 0; i < selectedViewsNumber; i++) {
        int cameraIdx = selectedViewsSubset[i];
        geometricCosts[i] = geometricCost_cu<T>(gs, samples, cameraIdx);
        // printf("in for i: %d, cameraIdx: %d, cost %f\n", i, cameraIdx, geometricCosts[i]);
    }

    sort_small(geometricCosts, selectedViewsNumber);
    //     for (int i = 0; i < selectedViewsNumber; i++) {
    //     printf("sorted in for i: %d, cost %f validNeighborsCount %d\n", i, geometricCosts[i], validNeighborsCount);
    // }
    // return 0.f;


    int numBest = validNeighborsCount; 
    if (gs.params->cost_comb == COMB_BEST_N)
        numBest = min(numBest, gs.params->n_best);

    float totalGeometricCost = 0.f;
    totalGeometricCost += numBest * geometricCostRefImage;
    for (int i = 0; i < numBest; i++) {
        totalGeometricCost += geometricCosts[i];
    }
    totalGeometricCost = totalGeometricCost / (2*(float)numBest);

    if(totalGeometricCost < -0.01f) {
        printf("geometric cost is less than 0! %f", totalGeometricCost);
    }

    float intensityCosts[20];
    const float intensityCostRefImage = intensityCost_cu<T>(gs, samples, 0);

    for (int i = 0; i < selectedViewsNumber; i++) {
        int cameraIdx = selectedViewsSubset[i];
        intensityCosts[i] = intensityCost_cu<T>(gs, samples, cameraIdx);
    }

    sort_small(intensityCosts, selectedViewsNumber);

    float totalIntensityCost = 0.f;
    for (int i = 0; i < numBest; i++) {
        totalIntensityCost += intensityCosts[i];
    }
    totalIntensityCost = totalIntensityCost / ((float)numBest);


    if(totalIntensityCost < -0.01f) {
        printf("inten cost is less than 0! %f", totalIntensityCost);
    }
    float alpha = 0.1;
    float cost = (1 - alpha) * totalGeometricCost + alpha * totalIntensityCost;
    float cost2 = getCombinedCosts_cu<T>(
        geometricCostRefImage,
        geometricCosts,
        intensityCostRefImage,
        intensityCosts,
        selectedViewsNumber,
        gs.params->cost_comb,
        gs.params->n_best,
        validNeighborsCount
    );

    // if (cost != cost2) {
    //     printf("WHAAAAT cost %f, cost2 %f\n", cost, cost2);
    // } else {
    //     printf("ok cost %f, cost2 %f\n", cost, cost2);
    // }
    // printf("geom %f int %f  total %f \n", geometricCost, intensityCost, cost);
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

    gs.lines->lineCost[center] = pmCostMultiview_cu<T>(
        gs, p, gs.lines->depth[center], gs.lines->unitDirection[center]);
    return;
}

template <typename T>
__device__ FORCEINLINE_GIPUMA void lineRefinement_cu(GlobalState &gs,
                                                         int2 referencePixel, const int it) {
    const int cols = gs.cameras->cols;

    float mind = gs.params->depthMin;
    float maxd = gs.params->depthMax;

    const int center = referencePixel.y * cols + referencePixel.x;
    curandState localState = gs.cs[center];

    const float maxdisp = gs.params->max_disparity / 2.0f;  // temp variable
    for (float deltaZ = maxdisp; deltaZ >= 0.01f; deltaZ = deltaZ / 10.0f) {    
        float3 newDir;
        rndUnitVectorSphereMarsaglia_cu(&newDir, &localState);


        float newDepth = curand_between(&localState, mind, maxd);

        float newCost = pmCostMultiview_cu<T>(gs, referencePixel, newDepth, newDir);

        if (newCost < gs.lines->lineCost[center]) {
            gs.lines->depth[center] = newDepth;
            gs.lines->unitDirection[center].x = newDir.x;
            gs.lines->unitDirection[center].y = newDir.y;
            gs.lines->unitDirection[center].z = newDir.z;
            gs.lines->lineCost[center] = newCost;
        }
    }
    
    return;
}

template <typename T>
__device__ FORCEINLINE_GIPUMA void spatialPropagation_cu(GlobalState &gs,
                                                         int2 referencePixel,
                                                         int2 otherPixel, const int it) {
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (otherPixel.x < 0 || otherPixel.x >= cols) return;
    if (otherPixel.y < 0 || otherPixel.y >= rows) return;

    const int center = referencePixel.y * cols + referencePixel.x;
    const int otherCenter = otherPixel.y * cols + otherPixel.x;

    // TODO: compute new depth
    float newDepth = gs.lines->depth[otherCenter];
    float3 newDir = gs.lines->unitDirection[otherCenter];

    float newCost = pmCostMultiview_cu<T>(gs, referencePixel, newDepth, newDir);

    if (newCost < gs.lines->lineCost[center]) {
        gs.lines->depth[center] = newDepth;
        gs.lines->unitDirection[center].x = newDir.x;
        gs.lines->unitDirection[center].y = newDir.y;
        gs.lines->unitDirection[center].z = newDir.z;
        gs.lines->lineCost[center] = newCost;
    }
    return;
}
template <typename T>
__global__ void gipuma_black_spatialPropClose_cu(GlobalState &gs, const int it) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if ((p.x  + p.y) % 2 == 0) {
        return;
    }
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;

    // Left
    int2 left;
    left.x = p.x - 1;
    left.y = p.y;
    // Up
    int2 up;
    up.x = p.x;
    up.y = p.y - 1;
    // Down
    int2 down;
    down.x = p.x;
    down.y = p.y + 1;
    // Right
    int2 right;
    right.x = p.x + 1;
    right.y = p.y;

    spatialPropagation_cu<T>(gs, p, left, it);
    spatialPropagation_cu<T>(gs, p, up, it);
    spatialPropagation_cu<T>(gs, p, down, it);
    spatialPropagation_cu<T>(gs, p, right, it);
}

template <typename T>
__global__ void gipuma_black_spatialPropFar_cu(GlobalState &gs, const int it) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if ((p.x  + p.y) % 2 == 0) {
        return;
    }
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;

    // Left
    int2 left;
    left.x = p.x - 5;
    left.y = p.y;
    // Up
    int2 up;
    up.x = p.x;
    up.y = p.y - 5;
    // Down
    int2 down;
    down.x = p.x;
    down.y = p.y + 5;
    // Right
    int2 right;
    right.x = p.x + 5;
    right.y = p.y;

    spatialPropagation_cu<T>(gs, p, left, it);
    spatialPropagation_cu<T>(gs, p, up, it);
    spatialPropagation_cu<T>(gs, p, down, it);
    spatialPropagation_cu<T>(gs, p, right, it);
}

template <typename T>
__global__ void gipuma_black_lineRefine_cu(GlobalState &gs, const int it) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if ((p.x  + p.y) % 2 == 0) {
        return;
    }
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;
    lineRefinement_cu<T>(gs, p, it);
}

template <typename T>
__global__ void gipuma_red_spatialPropClose_cu(GlobalState &gs, const int it) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if ((p.x  + p.y) % 2 == 1) {
        return;
    }
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;
    int2 left;
    left.x = p.x - 1;
    left.y = p.y;
    // Up
    int2 up;
    up.x = p.x;
    up.y = p.y - 1;
    // Down
    int2 down;
    down.x = p.x;
    down.y = p.y + 1;
    // Right
    int2 right;
    right.x = p.x + 1;
    right.y = p.y;

    spatialPropagation_cu<T>(gs, p, left, it);
    spatialPropagation_cu<T>(gs, p, up, it);
    spatialPropagation_cu<T>(gs, p, down, it);
    spatialPropagation_cu<T>(gs, p, right, it);
}

template <typename T>
__global__ void gipuma_red_spatialPropFar_cu(GlobalState &gs, const int it) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if ((p.x  + p.y) % 2 == 1) {
        return;
    }
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;

    // Left
    int2 left;
    left.x = p.x - 5;
    left.y = p.y;
    // Up
    int2 up;
    up.x = p.x;
    up.y = p.y - 5;
    // Down
    int2 down;
    down.x = p.x;
    down.y = p.y + 5;
    // Right
    int2 right;
    right.x = p.x + 5;
    right.y = p.y;

    spatialPropagation_cu<T>(gs, p, left, it);
    spatialPropagation_cu<T>(gs, p, up, it);
    spatialPropagation_cu<T>(gs, p, down, it);
    spatialPropagation_cu<T>(gs, p, right, it);
}

template <typename T>
__global__ void gipuma_red_lineRefine_cu(GlobalState &gs, const int it) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if ((p.x  + p.y) % 2 == 1) {
        return;
    }
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;
    lineRefinement_cu<T>(gs, p, it);
}

__global__ void gipuma_compute_disp(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    const int center = p.y * cols + p.x;
    float3 lineDirection = gs.lines->unitDirection[center];
    float3 direction_transformed;
    // Transform back direction to world coordinate
    matvecmul4(gs.cameras->cameras[REFERENCE].R_orig_inv, lineDirection,
               (&direction_transformed));

    // TODO: transform depth?
    // if (gs.lines->c[center] != MAXCOST)
    //     norm_transformed.w =
    //         getDisparity_cu(norm, norm.w, p, gs.cameras->cameras[REFERENCE]);
    // else
    //     norm_transformed.w = 0;
    gs.lines->unitDirection[center] = direction_transformed;
    return;
}

float getAverageCost(GlobalState &gs) {
    float c = 0.f;
    for (int i = 0; i < gs.cameras->rows; i++) {
        for (int j = 0; j < gs.cameras->cols; j++) {
            c += gs.lines->lineCost[i * gs.cameras->cols + j] / 1000.f;
        }
    }

    // printf("total cost 1000 * %f \n", c);
    float c2 = c / (gs.cameras->rows * gs.cameras->cols);
    return 1000.f * c2;
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

    printf("initial cost: %.8f \n", getAverageCost(gs));
    cudaEventRecord(start);
    printf("Iterations count %d", gs.params->iterations);
    for (int it =0;it<gs.params->iterations; it++) {
        // for (int it = 0; it < maxiter; it++) {
        printf("%d ", it + 1);
        // spatial propagation of 4 closest neighbors (1px up/down/left/right)
        gipuma_black_spatialPropClose_cu<T>
            <<<grid_size_initrand, block_size_initrand>>>(gs, it);
        cudaDeviceSynchronize();
        printf("cost black close: %.8f \n", getAverageCost(gs));


        // spatial propagation of 4 far away neighbors (5px up/down/left/right)
        gipuma_black_spatialPropFar_cu<T>
            <<<grid_size_initrand, block_size_initrand>>>(gs, it);
        cudaDeviceSynchronize();
        printf("cost black far: %.8f \n", getAverageCost(gs));
        // line refinement
        gipuma_black_lineRefine_cu<T>
            <<<grid_size_initrand, block_size_initrand>>>(gs, it);
        cudaDeviceSynchronize();
        printf("cost black refine: %.8f \n", getAverageCost(gs));

        // spatial propagation of 4 closest neighbors (1px up/down/left/right)
        gipuma_red_spatialPropClose_cu<T>
            <<<grid_size_initrand, block_size_initrand>>>(gs, it);
        cudaDeviceSynchronize();
        printf("cost red close: %.8f \n", getAverageCost(gs));


        // // spatial propagation of 4 far away neighbors (5px up/down/left/right)
        gipuma_red_spatialPropFar_cu<T>
            <<<grid_size_initrand, block_size_initrand>>>(gs, it);
        cudaDeviceSynchronize();
        printf("cost red far: %.8f \n", getAverageCost(gs));
        // line refinement
        gipuma_red_lineRefine_cu<T>
            <<<grid_size_initrand, block_size_initrand>>>(gs, it);
        cudaDeviceSynchronize();
        printf("cost red refine: %.8f \n", getAverageCost(gs));
    }
    printf("\n");

    // Transform directions to world space
    gipuma_compute_disp<<<grid_size_initrand, block_size_initrand>>>(gs);
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
