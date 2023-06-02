#pragma once

#include <iostream>
#include <fstream>


#include "fileIoUtils.h"
#include <stdexcept>

struct Line {
    // Z coord of 3D point in reference camera frame coord system
    float depth;
    // Normalized direction of line in world space
    Vec3f unitDirection;
};

// Given:   a pixel (u, v) and a Z coordinate d
// Returns: the 3D coordinates of the point in world space
Vec3f getPixelWorldCoord(int u, int v, const Mat_<float>& K, float d, const Mat_<float>& Rt);

// Given:   a 3D point and a 3D line
// Returns: another point that lies on the line
Vec3f getPointOn3DLine(const Vec3f& p, Line line);

// Given:   3D point world coordinates
// Returns: pixel coordinates
Vec2f projectPointToImage(const Vec3f& pointWorldCoord,  const Mat_<float>& K,  const Mat_<float>& Rt);


// Given:   3D point world coordinates and a line
// Returns: projection of the line in camera reference using Plucker coordinates
Vec2f projectLineIntoImagePlane(const Vec3f& point, const Line& line, const Mat_<float>& K, const Mat_<float>& Rt);


// Given:   pixel coordinates and a 3D line
// Returns: world coordinates of the point that lie on the 3D line 
Vec3f projectSamplePointIn3D(Vec2f samplePoint, const Vec3f& Point, const Line& line, const Mat_<float>& K, const Mat_<float>&  Rt);

void writeSamplesToImage(
    vector<Vec2f> samples,
    const char* outputFolder,
    const char* outputFileName,
    const char* inputImageFolder,
    string inputImageName
);

vector<vector<Vec2f>> samplePoints(
    size_t numViews,
    int k,
    float rk,
    vector<Camera> cameras,
    vector<string> imageFilenames
) {
    // TODO: use reference image height and width
    int width = 1100;
    int height = 1604;

    // TODO: remove duplication
    vector<Mat_<float>> Extrinsic(numViews); //Extrinsic 3*4
    vector<Mat_<float>> extrinsic(numViews); //Extrinsic 4*4, with the last row 0, 0, 0, 1

    for (int v = 0; v < numViews; v++) {
        Extrinsic[v] = Mat::zeros(3, 4, CV_32F);
        extrinsic[v] = Mat::eye(4, 4, CV_32F);
        for (int c = 0; c < 3; c++) {
            for (int d = 0; d < 3; d++) {
                Extrinsic[v](c, d) = cameras[v].R(c, d);
                extrinsic[v](c, d) = Extrinsic[v](c, d);
            }
        }
        for (int c = 0; c < 3; c++) {
            Extrinsic[v](c, 3) = cameras[v].t(c);;
            extrinsic[v](c, 3) = Extrinsic[v](c, 3);
        }
    }

    // Create a 3D line map with random depth values and unit vectors
    vector<vector<Line>> lineMap(height);
    // not so random Initialization
    for (int y = 0; y < height; ++y) {
        lineMap[y] = vector<Line>(width);
        for (int x = 0; x < width; ++x) {
            lineMap[y][x].depth = 1.1513861;
            
            lineMap[y][x].unitDirection << -0.10809, -0.709669, -0.696194;
            normalize(lineMap[y][x].unitDirection, lineMap[y][x].unitDirection);
        }
    }

    // TODO: use intrinsics of each image
    Mat_<float> K = cameras[0].K;

    const int u = 877;
    const int v = 1403;
    const int referenceImageIndex = 11;
    const int image1Index = 4;

    const Line line = lineMap[v][u];
    const Vec3f P1 = getPixelWorldCoord(u, v, K, line.depth, extrinsic[referenceImageIndex]);

    const Vec3f P2 = getPointOn3DLine(P1, line);
    const Vec2f pixel2 = projectPointToImage(P2, K, Extrinsic[referenceImageIndex]);

    const Vec2f pluckerCoord = projectLineIntoImagePlane(P1, line, K, Extrinsic[referenceImageIndex]);

    cout << "pluckerCoord: " << pluckerCoord.t() << endl;

    Vec2f lineInRef0UnitVector(
        u - pixel2[0],
        v - pixel2[1]
    );
    normalize(lineInRef0UnitVector, lineInRef0UnitVector);
    cout << "lineInRef0UnitVector normalized: " << lineInRef0UnitVector.t() << endl;

    Vec2f originalPixel(u, v);
    vector<Vec2f> samples;
    for (int i = -k/2; i <= k/2 ; i++) {
        Vec2f samplePixel = originalPixel + 2 * i * rk * lineInRef0UnitVector / k;
        samples.push_back(samplePixel);
    }
    

    vector<vector<Vec2f>> samplesInImages(numViews);
    for (int i = 0; i < numViews ; i++) {
        vector<Vec2f> samplesInImage;
        for (int j = 0; j < k ; j++) {
            Vec2f samplePixel = samples[j];

            const Vec3f pointWorldCoord(projectSamplePointIn3D(samplePixel, P1, line, K, Extrinsic[referenceImageIndex]));
            const Vec2f sampleImage1Coord = projectPointToImage(pointWorldCoord, K, Extrinsic[i]);
            samplesInImage.push_back(sampleImage1Coord);
        }
        samplesInImages[i] = samplesInImage;
        writeSamplesToImage(samplesInImage, "./res", imageFilenames[i].c_str() , "data/97_frame_00005",  imageFilenames[i]);
    }
    return samplesInImages;
}


Vec3f getPixelWorldCoord(int u, int v, const Mat_<float>& K, float d, const Mat_<float>& Rt) {
    Mat_<float> K_inv = K.inv();
    Mat_<float> Rt_inv = Rt.inv();
    Vec3f pixelHomogenCoord(u * d, v * d, d);

    Mat_<float> pointCameraCoord = K_inv * Mat(pixelHomogenCoord);
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

Vec3f getPointOn3DLine(const Vec3f& p, Line line) {
    return p + line.unitDirection;
}

Vec2f projectPointToImage(const Vec3f& pointWorldCoord,  const Mat_<float>& K,  const Mat_<float>& Rt) {

    const Vec4f pointWorldHomogenCoord(
        pointWorldCoord[0],
        pointWorldCoord[1],
        pointWorldCoord[2],
        1
    );

    const Mat_<float> pointCameraHomogenCoord = K * Rt * pointWorldHomogenCoord;
    const Vec2f pointCameraCoord(
        pointCameraHomogenCoord[0][0] / pointCameraHomogenCoord[2][0],
        pointCameraHomogenCoord[1][0] / pointCameraHomogenCoord[2][0]
    );

    // cout << endl<< endl<< endl;
    // cout << "In projectPointToImage" << endl;
    // cout << "K: " << K << endl;
    // cout << "Rt: " << Rt << endl;
    // cout << "pointWorldHomogenCoord: " << pointWorldHomogenCoord << endl;
    // cout << "pointCameraHomogenCoord: " << pointCameraHomogenCoord.transpose() << endl;
    // cout << "pointCameraCoord" << pointCameraCoord << endl;

    return pointCameraCoord;
}


Vec2f projectLineIntoImagePlane(const Vec3f& point, const Line& line, const Mat_<float>& K, const Mat_<float>& Rt) {
    Vec4f A(
        point[0],
        point[1],
        point[2],
        1
    );
    Vec4f B(
        point[0] + line.unitDirection[0],
        point[1] + line.unitDirection[1],
        point[2] + line.unitDirection[2],
        1
    );

    Matx<float, 4, 4> L = A * B.t() - B * A.t();
    Mat_<float> P = K * Rt;
    Mat_<float> res = P * L * P.t();
    Vec3f l(res(2, 1), res(0, 2), res(1, 0));
    Vec2f ll(l[1], -l[0]);

    normalize(ll, ll);

    return ll;
}


Vec3f projectSamplePointIn3D(Vec2f samplePoint, const Vec3f& Point, const Line& line, const Mat_<float>& K, const Mat_<float>&  Rt) {
    const Mat_<float> K_inv = K.inv();

    const Vec3f samplePointPixelHomogenCoord(
        samplePoint[0],
        samplePoint[1],
        1
    );

    const Mat_<float> samplePointCameraCoord = K_inv * samplePointPixelHomogenCoord;

    const float t2 = Rt(1, 3);
    const float t3 = Rt(2, 3);
    const float yPrim = samplePointCameraCoord(1, 0);
    const Vec3f P = Point;
    const Vec3f r2(
        Rt(1, 0),
        Rt(1, 1),
        Rt(1, 2)
    );

    const Vec3f r3(
        Rt(2, 0),
        Rt(2, 1),
        Rt(2, 2)
    );
    const Vec3f l = line.unitDirection;

    float lambda = (yPrim * r3.dot(Point) + yPrim * t3 - r2.dot(P) -t2) / (r2.dot(l) - yPrim * r3.dot(l));
    return Point + lambda * line.unitDirection;
}

void writeSamplesToImage(
    vector<Vec2f> samples,
    const char* outputFolder,
    const char* outputFileName,
    const char* inputImageFolder,
    string inputImageName
) {
    cout << "inputImageName" << inputImageName << endl;
    // TODO: do not re-read images
    Mat imageMatrix = readImage(inputImageFolder, inputImageName);

    for (unsigned int i = 0; i <= samples.size(); i++) {
        if (
            samples[i][1] < 0 ||
            samples[i][1] > imageMatrix.rows - 1 ||
            samples[i][0] < 0 ||
            samples[i][0] > imageMatrix.cols - 1
        ) {
            cout << "Note: sample is outside of image dimension (u, v): (" << samples[i][1] <<
                ", " << samples[i][1] << ")\n";
            continue;
        }
        Vec3b& color = imageMatrix.at<Vec3b>(samples[i][1], samples[i][0]);

        color[0] = 0;
        color[1] = 0;
        color[2] = 255;

        if (i == samples.size() / 2) {
            color[1] = 255;
        } else if (i > samples.size() / 2) {
            color[0] = 255;
            color[1] = 0;
            color[2] = 0;
        }
    }

    writeImageToFile(outputFolder, outputFileName, imageMatrix);
}






