#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace std;
//using namespace cv;


#define M_PI 3.14159265358979323846
#define STEP_SIZE 0.001f
#define RADIUS 0.001f

struct Points {
    cv::Vec3f position;
    cv::Vec3f direction;
    uint8_t r;
    uint8_t g;
    uint8_t b;

    bool operator==(const Points& other) const {
        // Compare the position and direction of two Points objects
        return position == other.position && direction == other.direction;
    }
};

typedef vector<Points> strand;

struct StrandComparator {
    bool operator()(const strand& a, const strand& b) const {
        if (a.size() != b.size()) {
            return a.size() > b.size();
        }
        // Compare the individual points within the strands
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i].position != b[i].position || a[i].direction != b[i].direction) {
                return a[i].position[0] < b[i].position[0];
            }
        }
        // If all points are equal, consider the strands as equal
        return false;
    }
};

typedef set<strand, StrandComparator> strand_set;


void read_point_cloud_binary_with_directions(std::string& file_path, std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>>& oriented_pointcloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    oriented_pointcloud.first = cloud;
    cout << "read 3D points and directions to file" << endl;
    std::ifstream file(file_path, std::ios::binary);

    long long int points_count;
    file.read(reinterpret_cast<char*>(&points_count), 8);

    cout << "points count: " << points_count << endl;

    std::vector<float> data(6 * points_count);
    file.read(reinterpret_cast<char*>(&data[0]), 6 * points_count * sizeof(float));
    for (size_t i = 0; i < data.size(); i += 6)
    {
        //std::cout << i + 5 << " " << data.size() << " " << points_count << std::endl;
        oriented_pointcloud.first->emplace_back(data.at(i), data.at(i + 1), data.at(i + 2));
        oriented_pointcloud.second.emplace_back(data.at(i + 3), data.at(i + 4), data.at(i + 5));
    }
    file.close();
}

void WritePLYFile(const std::string& filename, const vector<Points>& vertices) {
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

std::vector<Points> findNeighboringPoints(const vector<Points> pointcloud, const Points& point, float stepSize, float radius) {
    std::vector<Points> neighboringPoints;

    // Move the point along the line direction with step size
    Points movedPoint;
    movedPoint.position[0] = point.position[0] + stepSize * point.direction[0];
    movedPoint.position[1] = point.position[1] + stepSize * point.direction[1];
    movedPoint.position[2] = point.position[2] + stepSize * point.direction[2];

    // Search for neighboring points within the radius around the moved point
    for (const auto& neighborPoint : pointcloud) {
        double distance = std::sqrt(std::pow(neighborPoint.position[0] - movedPoint.position[0], 2) +
            std::pow(neighborPoint.position[1] - movedPoint.position[1], 2) +
            std::pow(neighborPoint.position[2] - movedPoint.position[2], 2));

        //not treat the moved point as a neighboring point
        if (distance <= radius && neighborPoint.position != movedPoint.position && neighborPoint.position != point.position) {
            neighboringPoints.push_back(neighborPoint);
        }
    }


    return neighboringPoints;
}

float AngularDifference(const Points& point_1, const Points& point_2) {
    float dot_product = point_1.direction.dot(point_2.direction);
    float point1_len = cv::norm(point_1.direction);
    float point2_len = cv::norm(point_2.direction);

    float cosinetheta = dot_product / (point1_len * point2_len);
    float theta = std::acos(cosinetheta);

    // Convert the angle to degrees
    float thetadegrees = theta * 180.0 / M_PI;
    return thetadegrees;
}

Points AveragePoint(const Points& point, const vector<Points>& neighbor_points, float step_size) {
    Points average_point;
    int neighbors_num = neighbor_points.size();
    int valid_points = 0;
    for (int i = 0; i < neighbors_num; i++) {
        if (AngularDifference(neighbor_points[i], point) < 30) {
            average_point.position += neighbor_points[i].position;
            average_point.direction += neighbor_points[i].direction;
            valid_points++;
        }
    }
    cout << "The number of valid points are " << valid_points << "\n";

    //check the number of valid points
    if (valid_points != 0) {
        average_point.position /= valid_points;
        average_point.direction /= valid_points;
    }
    //if no valid points, the average point is equal to the moved point
    else {
        Points movedPoint;
        movedPoint.position[0] = point.position[0] + step_size * point.direction[0];
        movedPoint.position[1] = point.position[1] + step_size * point.direction[1];
        movedPoint.position[2] = point.position[2] + step_size * point.direction[2];
        average_point.position[0] = movedPoint.position[0];
        average_point.position[1] = movedPoint.position[1];
        average_point.position[2] = movedPoint.position[2];
        average_point.direction[0] = point.direction[0];
        average_point.direction[1] = point.direction[1];
        average_point.direction[2] = point.direction[2];
    }
    
    return average_point;
}

float distance(const Points& p1, const Points& p2) {
    float dx = p1.position[0] - p2.position[0];
    float dy = p1.position[1] - p2.position[1];
    float dz = p1.position[2] - p2.position[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}


vector<Points> removePointsWithinRadius(std::vector<Points>& pointcloud, const std::vector<Points>& strand, float radius) {
    std::vector<Points> removedPointCloud;

    for (const Points& p : pointcloud) {
        bool withinRadius = false;
        for (const Points& strandPoint : strand) {
            if (distance(p, strandPoint) <= radius) {
                withinRadius = true;
                break;
            }
        }
        if (withinRadius) {
            removedPointCloud.push_back(p);
        }
    }

    return removedPointCloud;
}

void removePointsFromPointCloud(std::vector<Points>& pointcloud, const std::vector<Points>& strand_1) {
    pointcloud.erase(std::remove_if(pointcloud.begin(), pointcloud.end(),
        [&](const Points& p) {
            return std::find(strand_1.begin(), strand_1.end(), p) != strand_1.end();
        }), pointcloud.end());
}


void WriteStrandPLYFile(const std::string& filename, const strand_set& strandSet) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int vertexSize = 0;
    for (const auto& strand : strandSet) {
        vertexSize += strand.size();
    }

    // Write the PLY header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << vertexSize << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // Write the vertex data
    for (const auto& strand : strandSet) {
        for (const auto& point : strand) {
            file.write(reinterpret_cast<const char*>(&point.position[0]), sizeof(point.position[0]));
            file.write(reinterpret_cast<const char*>(&point.position[1]), sizeof(point.position[1]));
            file.write(reinterpret_cast<const char*>(&point.position[2]), sizeof(point.position[2]));
            file.write(reinterpret_cast<const char*>(&point.r), sizeof(point.r));
            file.write(reinterpret_cast<const char*>(&point.g), sizeof(point.g));
            file.write(reinterpret_cast<const char*>(&point.b), sizeof(point.b));
        }
    }



    file.close();

    std::cout << "PLY file written: " << filename << std::endl;
}


//write the strandset to a file to test hair growing 
void OutputStrandstoFile(const std::string& filename, const strand_set& strandSet) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto& strand : strandSet) {
        for (const auto& point : strand) {
            file << point.position[0] << " " << point.position[1] << " " << point.position[2] << " " << point.direction[0] << " " << point.direction[1] << " " << point.direction[2] << "\n";
        }
    }

    file.close();

    std::cout << "PLY file written: " << filename << std::endl;
}

int main() {
    //std::string file_path = "line_cloud_output_new_weights_1.dat";
    std::string file_path = "line_cloud_output_new_weights.dat";
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>> oriented_pointcloud;

    read_point_cloud_binary_with_directions(file_path, oriented_pointcloud);
    cout << "The output pointcloud is read successfully!" << "\n";

    // Access the point cloud and directions
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = oriented_pointcloud.first;
    std::vector<Eigen::Vector3f>& directions = oriented_pointcloud.second;

    pcl::io::savePLYFileASCII("fused.ply", *(oriented_pointcloud.first));
    cout << "Get the fused data" << "\n";

    vector<Points> pointcloud;
    // Make sure the sizes match
    size_t numPoints = cloud->size();
    size_t numDirections = directions.size();
    if (numPoints != numDirections) {
    }

    // Convert point cloud and directions into Points structure
    for (size_t i = 0; i < numPoints; ++i) {
        const pcl::PointXYZ& point = (*cloud)[i];
        const Eigen::Vector3f& dir = directions[i];

        Points p;
        p.position = cv::Vec3f(point.x, point.y, point.z);
        p.direction = cv::Vec3f(dir[0], dir[1], dir[2]);

        pointcloud.push_back(p);
    }
    WritePLYFile("allpoints.ply", pointcloud);


    //start strands generation

    //create a strand_set
    strand_set strands_set;
    //neighboring points in the positive direction
    int n = 1;
    //neighboring points in the negative direction
    int m = 1;
    //number of iterations
    int num = 0;
    //define a counter to avoid getting stuck at two points, 
    //if one iteration takes too much time, it is more likely jumping between two points
    int counter_positive = 0;
    int counter_negative = 0;
    while (!pointcloud.empty()) {
        cout << "iteration " << num << ":\n";
        vector<Points> strand_1;
        Points current_point = pointcloud[0];
        strand_1.push_back(current_point);
        cout << "The current point is " << current_point.position << "\n";

        //positive direction
        //if there is neighbors in the moving direction
        n = 1;
        counter_positive = 0;
        counter_negative = 0;
        while (n != 0 && counter_positive <= 100) {

            //take the first point in the pointcloud
            //find its neighbor points within a radius 
            cout << "The positive direction\n";
            vector<Points> neighbor_points = findNeighboringPoints(pointcloud, current_point, STEP_SIZE, RADIUS);
            n = neighbor_points.size();
            if (n == 0) {
                break;
            }

            cout << "the number of neighboring point is " << neighbor_points.size() << "\n";
            //cout << "the neighboring point is " << neighbor_points[0].position << "\n";
            //calculate the average point and add it to the strand
            Points average_point = AveragePoint(current_point, neighbor_points, STEP_SIZE);
            cout << "The average point is " << average_point.position << "\n";
            strand_1.push_back(average_point);
            current_point = average_point;
            cout << "Get the average new point\n";
            counter_positive++;
        }

        //search its negative direction of the original point
        current_point = pointcloud[0];
        while (m != 0 && counter_negative <= 100) {
            cout << "The negative direction;\n";
            vector<Points> neighbor_points = findNeighboringPoints(pointcloud, current_point, -STEP_SIZE, RADIUS);
            m = neighbor_points.size();

            //if there is no neighboring point stop this iteration
            if (m == 0) {
                break;
            }

            cout << "the number of neighboring point is " << neighbor_points.size() << "\n";
            //calculate the average point and add it to the strand
            Points average_point = AveragePoint(current_point, neighbor_points, STEP_SIZE);
            cout << "The average point is " << average_point.position << "\n";
            strand_1.push_back(average_point);
            current_point = average_point;
            cout << "Get the average new point\n";
            counter_negative++;
        }

        cout << "The size of the strand is " << strand_1.size() << "\n";
        cout << "The point 0 is " << pointcloud[0].position << "\n";
        vector<Points> strand_neighbors = removePointsWithinRadius(pointcloud, strand_1, RADIUS);
        removePointsFromPointCloud(pointcloud, strand_1);
        removePointsFromPointCloud(pointcloud, strand_neighbors);
        cout << "The size of the pointcloud is " << pointcloud.size() << "\n";
        num++;
        //if the strand only contains one point, it can be served as noise
        if (strand_1.size() > 1) {
            strands_set.insert(strand_1);
            cout << "Get the " << num << "th  strand\n";
        }
    }
    cout << "the number of the strand is " << strands_set.size() << "\n";
    cout << "The final size of the pointcloud is" << pointcloud.size() << "\n";


    WriteStrandPLYFile("all_output_strand.ply", strands_set);
    
    return 0;
}
