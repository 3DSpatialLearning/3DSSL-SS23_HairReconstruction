#pragma once

#include <iostream>
#include <fstream>

#include "fileIoUtils.h"
#include "line.h"

// Given:   a pixel (u, v) and a Z coordinate d
// Returns: the 3D coordinates of the point in world space
Vec3f getPixelWorldCoord(Vec2i pixel, float d, const Mat_<float>& K_inv, const Mat_<float>& Rt_inv) {
    Vec3f pixelHomogenCoord(pixel[0] * d, pixel[1] * d, d);

    Mat_<float> pointCameraCoord = K_inv * pixelHomogenCoord;
    Vec4f pointCameraHomogenCoord(
        pointCameraCoord[0][0],
        pointCameraCoord[1][0],
        pointCameraCoord[2][0],
        1.f
    );

    Mat_<float> pointWorldHomogenCoord = Rt_inv * pointCameraHomogenCoord;
    Vec3f pointWorldCoord(
        pointWorldHomogenCoord[0][0],
        pointWorldHomogenCoord[1][0],
        pointWorldHomogenCoord[2][0]
    );

    return pointWorldCoord;
}

// Given:   3D point world coordinates
// Returns: pixel coordinates
Vec2f projectPointToImage(const Vec3f& pointWorldCoord, const Mat_<float>& P) {
    const Vec4f pointWorldHomogenCoord(
        pointWorldCoord[0],
        pointWorldCoord[1],
        pointWorldCoord[2],
        1
    );

    const Mat_<float> pointCameraHomogenCoord = P * pointWorldHomogenCoord;
    const Vec2f pointCameraCoord(
        pointCameraHomogenCoord[0][0] / pointCameraHomogenCoord[2][0],
        pointCameraHomogenCoord[1][0] / pointCameraHomogenCoord[2][0]
    );

    return pointCameraCoord;
}


// Given:   3D point world coordinates
// Returns: 3D camera coordinates
Vec3f pointWorldCoordToCameraCoord(const Vec3f& pointWorldCoord, const Mat_<float>& Rt) {
    Vec4f pointWorldCoordHomo(
        pointWorldCoord[0],
        pointWorldCoord[1],
        pointWorldCoord[2],
        1
    );

    Mat_<float> pointCameraCoordMat = Rt * pointWorldCoordHomo;

    const Vec3f pointCameraCoord(
        pointCameraCoordMat[0][0],
        pointCameraCoordMat[1][0],
        pointCameraCoordMat[2][0]
    );

    return pointCameraCoord;
}



