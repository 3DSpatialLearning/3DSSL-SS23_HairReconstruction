#pragma once

#include <fstream>
#include <iostream>

#include "fileIoUtils.h"
#include "line.h"
#include "projectionUtils.h"
#include <stdexcept>

// Given:   a 3D point and a 3D line
// Returns: another point that lies on the line
Vec3f getPointOn3DLine(const Vec3f& p, Line line);

// Given:   3D point world coordinates and a line
// Returns: projection of the line in camera reference using Plucker coordinates
Vec2f projectLineIntoImagePlane(const Vec3f& point, const Line& line,
                                const Mat_<float>& K, const Mat_<float>& Rt);

// Given:   pixel coordinates and a 3D line
// Returns: world coordinates of the point that lie on the 3D line
Vec3f projectSamplePointIn3D(Vec2f samplePoint, const Vec3f& Point,
                             const Line& line, const Mat_<float>& K_inv,
                             const Mat_<float>& Rt);

void writeSamplesToImage(vector<Vec2f> samples, const char* outputFolder,
                         string outputFileName, string inputImageFolder,
                         string inputImageName);

vector<vector<Vec2f>> samplePoints(
    size_t numViews,
    int k,
    float rk,
    Vec2i pixelCoord,
    Line line,
    vector<Camera> cameras,
    vector<string> imageFilenames,
    string images_folder,
    const char* lineProjectionImagesFolder,
    bool writeSamplesInImages = false
) {
    // referenceImage is always the first image passed as an argument to ./gipuma
    int referenceImageIndex = 0;

    const int u = pixelCoord[0];
    const int v = pixelCoord[1];
    const Vec3f P1 = getPixelWorldCoord(
        pixelCoord, line.depth, cameras[referenceImageIndex].K_inv,
        cameras[referenceImageIndex].Rt_extended_inv);

    const Vec3f P2 = getPointOn3DLine(P1, line);
    const Vec2f pixel2 =
        projectPointToImage(P2, cameras[referenceImageIndex].P);

    const Vec2f pluckerCoord =
        projectLineIntoImagePlane(P1, line, cameras[referenceImageIndex].K,
                                  cameras[referenceImageIndex].Rt);

    // cout << "pluckerCoord: " << pluckerCoord.t() << endl;

    Vec2f lineInRef0UnitVector(
        u - pixel2[0],
        v - pixel2[1]
    );
    normalize(lineInRef0UnitVector, lineInRef0UnitVector);

    Vec2f originalPixel(u, v);
    vector<Vec2f> samples;
    for (int i = -k/2; i <= k/2 ; i++) {
        Vec2f samplePixel = originalPixel + 2 * i * rk * lineInRef0UnitVector / k;
        samples.push_back(samplePixel);
    }

    vector<vector<Vec2f>> samplesInImages(numViews);
    for (int i = 0; i < numViews; i++) {
        vector<Vec2f> samplesInImage;
        for (int j = 0; j < k; j++) {
            Vec2f samplePixel = samples[j];

            const Vec3f pointWorldCoord = projectSamplePointIn3D(samplePixel, P1, line, cameras[referenceImageIndex].K_inv, cameras[referenceImageIndex].Rt);
            const Vec2f sampleImage1Coord = projectPointToImage(pointWorldCoord, cameras[i].P);
            samplesInImage.push_back(sampleImage1Coord);
        }
        samplesInImages[i] = samplesInImage;
    }

    if (writeSamplesInImages) {
        for (int i = 0; i < numViews ; i++) {
            writeSamplesToImage(
                samplesInImages[i],
                lineProjectionImagesFolder,
                imageFilenames[i],
                images_folder,
                imageFilenames[i]
            );
        }
    }

    return samplesInImages;
}

Vec3f getPointOn3DLine(const Vec3f& p, Line line) {
    return p + line.unitDirection;
}

Vec2f projectLineIntoImagePlane(
    const Vec3f& point,
    const Line& line,
    Mat_<float>& K,
    const Mat_<float>& Rt
) {
    Vec4f A(point[0], point[1], point[2], 1);
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

Vec3f projectSamplePointIn3D(Vec2f samplePoint, const Vec3f& Point,
                             const Line& line, const Mat_<float>& K_inv,
                             const Mat_<float>& Rt) {
    const Vec3f samplePointPixelHomogenCoord(samplePoint[0], samplePoint[1], 1);

    const Mat_<float> samplePointCameraCoord = K_inv * samplePointPixelHomogenCoord;

    const float t1 = Rt(0, 3);
    const float t2 = Rt(1, 3);
    const float t3 = Rt(2, 3);

    const float xPrim = samplePointCameraCoord(0, 0);
    const float yPrim = samplePointCameraCoord(1, 0);
    const Vec3f P = Point;

    const Vec3f r1(Rt(0, 0), Rt(0, 1), Rt(0, 2));
    const Vec3f r2(Rt(1, 0), Rt(1, 1), Rt(1, 2));
    const Vec3f r3(Rt(2, 0), Rt(2, 1), Rt(2, 2));

    const Vec3f l = line.unitDirection;

    float lambda = 0;
    const float d = r2.dot(l) - yPrim * r3.dot(l);
    const float n = yPrim * r3.dot(Point) + yPrim * t3 - r2.dot(P) - t2;
    const float d2 = (r1.dot(l) - xPrim * r3.dot(l));
    const float n2 = xPrim * r3.dot(Point) + xPrim * t3 - r1.dot(P) - t1;

    if (abs(d) < 0.001f) {
        lambda = n2 / d2;
    } else {
        lambda = n / d;
    }

    return Point + lambda * line.unitDirection;
}

void writeSamplesToImage(
    vector<Vec2f> samples,
    const char* outputFolder,
    string outputFileName,
    string inputImageFolder,
    string inputImageName
) {
    // TODO: do not re-read images
    Mat imageMatrix = readImage(inputImageFolder, inputImageName);
    for (unsigned int i = 0; i < samples.size(); i++) {
        if (
            samples[i][1] < 0 ||
            samples[i][1] > imageMatrix.rows - 1 ||
            samples[i][0] < 0 ||
            samples[i][0] > imageMatrix.cols - 1
        ) {
            cout << "Note: sample is outside of image dimension (u, v): (" << samples[i][0] <<
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
    outputFileName = outputFileName.substr(0, outputFileName.size() - 4);
    writeImageToFile(outputFolder, outputFileName.c_str(), imageMatrix);
}
