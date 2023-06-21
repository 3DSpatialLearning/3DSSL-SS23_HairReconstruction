#pragma once 

struct Line {
    // Z coord of 3D point in reference camera frame coord system
    float depth;
    // Normalized direction of line in world space
    Vec3f unitDirection;
};