#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include<string>

using namespace std;

//point in the point cloud
struct Point {
    cv::Vec3f position;
    cv::Vec3f direction;
    uint8_t r, g, b;
};

//create a pointcloud of a line and with some normal distributed noise
//input: the length of the line, the stddev of the noise
//output: a pointcloud
std::vector<Point> CreatePointClouds(float length, float noise) {
    std::random_device rd;
    std::mt19937 gen(rd());
    //uniform distribution
    //std::uniform_real_distribution<float> dis(-noise, noise);
    //Gaussian Distribution
    float mean = 0.0f;
    float stddev = noise;

    // Create a normal distribution with the desired mean and standard deviation
    std::normal_distribution<float> dis(mean, stddev);

    // Generate a random number from the Gaussian distribution
    float value = dis(gen);

    std::vector<Point> pointClouds;
    // Line 1
    float line1DirectionX = 1.0f;
    float line1DirectionY = 0.0f;
    float line1DirectionZ = 0.0f;

    for (float t = -length; t <= length; t += 0.01f) {
        float noiseX = dis(gen);
        float noiseY = dis(gen);
        float noiseZ = dis(gen);

        Point point;
        point.position[0] = t * line1DirectionX + noiseX;
        point.position[1] = t * line1DirectionY + noiseY;
        point.position[2] = t * line1DirectionZ + noiseZ;
        point.direction[0] = 1.0;
        point.direction[1] = 1.0;
        point.direction[2] = 1.0;
        point.r = 0.0;
        point.g = 0.0;
        point.b = 255.0;

        pointClouds.push_back(point);
    }
    return pointClouds;
}

//visualize the direction of each point by adding extra points along the direction
//input: a single point
//output: a line shows the direction of the point
std::vector<Point> DrawDirection(const Point& point) {
    std::vector<Point> line_points(20);
    for (float i = 1; i < 20; i++) {
        if (i != 0) {
            line_points[i].position[0] = point.position[0] + std::pow(-1, i) * (0.02) * i * point.direction[0];
            line_points[i].position[1] = point.position[1] + std::pow(-1, i) * (0.02) * i * point.direction[1];
            line_points[i].position[2] = point.position[2] + std::pow(-1, i) * (0.02) * i * point.direction[2];
            line_points[i].r = 255.0;
            line_points[i].g = 0.0;
            line_points[i].b = 0.0;
        }  
    }
    return line_points;
}

//Write the points into the a output .ply file
//input: pointcloud and filename
//output: .ply file
void WritePLYFile(const std::string& filename, const vector<Point>& vertices) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write the PLY header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // Write the vertex data
    for (const auto& vertex : vertices) {
        file.write(reinterpret_cast<const char*>(&vertex.position[0]), sizeof(vertex.position[0]));
        file.write(reinterpret_cast<const char*>(&vertex.position[1]), sizeof(vertex.position[1]));
        file.write(reinterpret_cast<const char*>(&vertex.position[2]), sizeof(vertex.position[2]));
        file.write(reinterpret_cast<const char*>(&vertex.r), sizeof(vertex.r));
        file.write(reinterpret_cast<const char*>(&vertex.g), sizeof(vertex.g));
        file.write(reinterpret_cast<const char*>(&vertex.b), sizeof(vertex.b));
    }

    file.close();

    std::cout << "PLY file written: " << filename << std::endl;
}

//call this function for the visualization
//input: pointcloud
//output: .ply file
void VisualizeLineMap(vector<Point> pointcloud) {
    int points_num = pointcloud.size();
    for (int j = 0; j < points_num; j++) {
        vector<Point> line = DrawDirection(pointcloud[j]);
        pointcloud.insert(pointcloud.end(), line.begin(), line.end());
    }
    WritePLYFile("visualization_result.ply", pointcloud);
}

int main() {
    //test
    //create a pointcloud
    vector<Point> pointcloud = CreatePointClouds(10.0f, 0.6f);
    //write the original pointcloud without direction(not neccessary)
    WritePLYFile("points.ply", pointcloud);
    //visualize the line map with all the positions and directions
    VisualizeLineMap(pointcloud);
    return 0;
}








