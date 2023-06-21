#pragma once

#include <iostream>
#include <fstream>

#include "line.h"
#include "projectionUtils.h"
#include <stdexcept>
#include <iomanip> 

const float EPSILON = 0.0001;

// Returns new line parameters for pixel1 based on the current
// line of pixel2.
// Using: https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
Line refineLine(
    Vec2i pixel1,
    Line line1,
    Vec2i pixel2,
    Line line2,
    Vec3f CameraCenter,
    Mat_<float> K_inv,
    Mat_<float> Rt_inv,
    Mat_<float> Rt
) {
    Vec3f pWorldCoord = getPixelWorldCoord(pixel1, line1.depth, K_inv, Rt_inv);

    Vec3f l1 = CameraCenter - pWorldCoord;
    l1 = normalize(l1);

    Vec3f l2 = line2.unitDirection;

    Vec3f l3 = l1.cross(l2);
    l3 = normalize(l3);

    if (l3[0] < EPSILON && l3[1] < EPSILON && l3[2] < EPSILON) {
        cout << "Lines are parallel, returning same line direction";
        return line1;
    }

    // Left hand side
    Mat_<float> lhs(3, 3);
    lhs[0][0] = l1[0];
    lhs[1][0] = l1[1];
    lhs[2][0] = l1[2];
    
    lhs[0][1] = -l2[0];
    lhs[1][1] = -l2[1];
    lhs[2][1] = -l2[2];

    lhs[0][2] = l3[0];
    lhs[1][2] = l3[1];
    lhs[2][2] = l3[2];

    Vec3f point1 = pWorldCoord;
    Vec3f point2 = getPixelWorldCoord(pixel2, line2.depth, K_inv, Rt_inv);
    Vec3f rhs = point2 - point1;

    Mat_<float> params = lhs.inv() * rhs;

    Vec3f new3DPoint = point1 + params[0][0] * l1;

    Vec3f newCameraCoord = pointWorldCoordToCameraCoord(new3DPoint, Rt);
    float newDepth = newCameraCoord[2];

    Line updatedLine;
    updatedLine.depth = newDepth;
    updatedLine.unitDirection = line2.unitDirection;
    
    return updatedLine;
}