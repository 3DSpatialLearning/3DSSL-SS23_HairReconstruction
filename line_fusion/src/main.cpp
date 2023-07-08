#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <random>
#include <LineFusion.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#define EXPORT_POINTCLOUD 1

void read_point_cloud_binary_with_directions(std::string &file_path, std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>> &oriented_pointcloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    oriented_pointcloud.first = cloud;
    cout << "read 3D points and directions to file" << endl;
    std::ifstream file(file_path, std::ios::binary);

    long int points_count;
    file.read(reinterpret_cast<char *>(&points_count), sizeof(long int));

    cout << "points count: " << points_count << endl;

    std::vector<float> data(6 * points_count);
    file.read(reinterpret_cast<char *>(&data[0]), 6 * points_count * sizeof(float));
    oriented_pointcloud.first->resize(points_count);
    oriented_pointcloud.second.resize(points_count);
    for (size_t i = 0; i < data.size(); i += 6)
    {
        std::cout << i + 5 << " " << data.size() << " " << points_count << std::endl;
        oriented_pointcloud.first->emplace_back(data.at(i), data.at(i + 1), data.at(i + 2));
        oriented_pointcloud.second.emplace_back(data.at(i + 3), data.at(i + 4), data.at(i + 5));
    }
    file.close();
}
void visualize_pointclouds_with_orientations(std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>> oriented_pointcloud)
{
    pcl::visualization::PCLVisualizer viewer("Point Cloud with Orientations");

    // Add point cloud to the viewer
    viewer.addPointCloud(oriented_pointcloud.first, "point_cloud");

    // Add lines representing orientations
    for (size_t i = 0; i < oriented_pointcloud.first->size(); i++)
    {
        const pcl::PointXYZ &point = oriented_pointcloud.first->at(i);
        const Eigen::Vector3f &direction = oriented_pointcloud.second[i];

        pcl::PointXYZ start_point(point.x, point.y, point.z);

        float factor = 0.2;
        pcl::PointXYZ end_point(
            point.x + factor * direction[0],
            point.y + factor * direction[1],
            point.z + factor * direction[2]);

        // Create a unique ID for each line
        std::string line_id = "line_" + std::to_string(i);

        // Add the line segment to the viewer
        viewer.addLine(start_point, end_point, line_id);
    }

    viewer.addCoordinateSystem();
    // Set the viewer's background color
    viewer.setBackgroundColor(0.0, 0.0, 0.0);

    // Spin the viewer
    viewer.spin();
}

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>> create_random_pointcloud(int num_points, float noise_mean, float noise_stddev, const std::string &file_name)
{

    // Create point cloud object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<Eigen::Vector3f> directions; // Vector to store directions

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(noise_mean, noise_stddev);
    std::uniform_real_distribution<float> angle_dist(-1, 1);    // Distribution for random angles in radians (0 to π)
    std::uniform_real_distribution<float> angle_dist_pos(0, 1); // Distribution for random angles in radians (0 to π)

    // Generate points for the first line
    for (int i = 0; i < num_points; ++i)
    {
        float x = -1.0 + (2.0 / num_points) * i; // X values range from -1 to 1
        float y = 1.0 + noise_dist(gen);         // Y values are constant for the first line
        float z = noise_dist(gen);
        float angle = angle_dist(gen); // Random angle 1 in radians (0 to π)
        float a, b, c;
        a = 0.4;
        b = 0.6;
        c = 0;

        directions.emplace_back(a, b, c);
        cloud->push_back(pcl::PointXYZ(x, y, z)); // Add point to the point cloud
        // if (angle == M_PI)
        // else
        //     directions.emplace_back(std::cos(angle), std::sin(angle), 0.0); // Convert angle to direction vector
    }

    // Generate points for the second line
    // for (int i = 0; i < num_points; ++i)
    // {
    //     float x = -1.0 + (2.0 / num_points) * i; // X values range from -1 to 1
    //     float y = -1.0 - noise_dist(gen);        // Y values are constant for the second line
    //     float z = noise_dist(gen);
    //     float angle = angle_dist(gen); // Random angle 1 in radians (0 to π)

    //     cloud->push_back(pcl::PointXYZ(x, y, z)); // Add point to the point cloud
    //     if (angle == M_PI)
    //         directions.emplace_back(1.0, 0.0, 0.0);
    //     else
    //         directions.emplace_back(std::cos(angle), std::sin(angle), 0.0); // Convert angle to direction vector
    // }

#if EXPORT_POINTCLOUD
    // Save the point cloud to a PCD file
    pcl::io::savePLYFileASCII(file_name, *cloud);
#endif

    return std::make_pair(cloud, directions);
}

int main()
{
    // Define parameters
    // int num_points = 1000;     // Number of points in each line
    // float noise_mean = 0.0;   // Mean of the noise
    // float noise_stddev = 0.2; // Standard deviation of the noise
    // std::string file_name{"/home/usamex/3dsl/line_fusion/pointclouds/noisy_lines.ply"};
    //  = create_random_pointcloud(num_points, noise_mean, noise_stddev, file_name);
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>> hair_pointcloud;
    std::string file_name{"/home/usamex/3dsl/line_fusion/pointclouds/line_cloud.dat"};
    read_point_cloud_binary_with_directions(file_name, hair_pointcloud);
    LineFusion line_fusion(hair_pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3f>> fused_line_cloud;
    fused_line_cloud.first = cloud;
    line_fusion.line_fusion(fused_line_cloud);
    std::string file_name_fused{"/home/usamex/3dsl/line_fusion/pointclouds/fused_lines.ply"};

    pcl::io::savePLYFileASCII(file_name_fused, *(fused_line_cloud.first));
    visualize_pointclouds_with_orientations(hair_pointcloud);
    visualize_pointclouds_with_orientations(fused_line_cloud);

    return 0;
}
