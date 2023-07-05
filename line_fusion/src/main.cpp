#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <random>
#include <LineFusion.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#define EXPORT_POINTCLOUD 1

void visualize_pointclouds_with_orientations(std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3d>> oriented_pointcloud)
{
    pcl::visualization::PCLVisualizer viewer("Point Cloud with Orientations");

    // Add point cloud to the viewer
    viewer.addPointCloud(oriented_pointcloud.first, "point_cloud");

    // Add lines representing orientations
    for (size_t i = 0; i < oriented_pointcloud.first->size(); i++)
    {
        const pcl::PointXYZ &point = oriented_pointcloud.first->at(i);
        const Eigen::Vector3d &direction = oriented_pointcloud.second[i];

        pcl::PointXYZ start_point(point.x, point.y, point.z);

        float factor = 0.2;
        pcl::PointXYZ end_point(
            point.x + factor * direction[0],
            point.y + factor * direction[1],
            point.z);

        // Create a unique ID for each line
        std::string line_id = "line_" + std::to_string(i);

        // Add the line segment to the viewer
        viewer.addLine(start_point, end_point, line_id);
    }

    // Set the viewer's background color
    viewer.setBackgroundColor(0.0, 0.0, 0.0);

    // Spin the viewer
    viewer.spin();
}

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3d>> create_random_pointcloud(int num_points, double noise_mean, double noise_stddev, const std::string &file_name)
{

    // Create point cloud object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<Eigen::Vector3d> directions; // Vector to store directions

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist(noise_mean, noise_stddev);
    std::uniform_real_distribution<double> angle_dist(0.0, M_PI); // Distribution for random angles in radians (0 to π)

    // Generate points for the first line
    for (int i = 0; i < num_points; ++i)
    {
        double x = -1.0 + (2.0 / num_points) * i; // X values range from -1 to 1
        double y = 1.0 + noise_dist(gen);         // Y values are constant for the first line
        double z = noise_dist(gen);
        double angle = angle_dist(gen); // Random angle 1 in radians (0 to π)

        cloud->push_back(pcl::PointXYZ(x, y, z)); // Add point to the point cloud
        if (angle == M_PI)
            directions.emplace_back(0.0, 1.0, 0.0);
        else
            directions.emplace_back(0, std::cos(angle), std::sin(angle)); // Convert angle to direction vector
    }

    // Generate points for the second line
    for (int i = 0; i < num_points; ++i)
    {
        double x = -1.0 + (2.0 / num_points) * i; // X values range from -1 to 1
        double y = -1.0 - noise_dist(gen);        // Y values are constant for the second line
        double z = noise_dist(gen);
        double angle = angle_dist(gen); // Random angle 1 in radians (0 to π)

        cloud->push_back(pcl::PointXYZ(x, y, z)); // Add point to the point cloud
        if (angle == M_PI)
            directions.emplace_back(1.0, 0.0, 0.0);
        else
            directions.emplace_back(std::cos(angle), std::sin(angle), 0.0); // Convert angle to direction vector
    }

#if EXPORT_POINTCLOUD
    // Save the point cloud to a PCD file
    pcl::io::savePLYFileASCII(file_name, *cloud);
#endif

    return std::make_pair(cloud, directions);
}

int main()
{
    // Define parameters
    int num_points = 10000;   // Number of points in each line
    double noise_mean = 0.0;   // Mean of the noise
    double noise_stddev = 0.1; // Standard deviation of the noise
    std::string file_name{"/home/usamex/3dsl/usame_orient/pointclouds/noisy_lines.ply"};
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3d>> noisy_oriented_pointcloud = create_random_pointcloud(num_points, noise_mean, noise_stddev, file_name);

    LineFusion line_fusion(noisy_oriented_pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<Eigen::Vector3d>> fused_line_cloud;
    fused_line_cloud.first = cloud;
    line_fusion.line_fusion(fused_line_cloud);
    std::string file_name_fused{"/home/usamex/3dsl/usame_orient/pointclouds/fused_lines.ply"};

    pcl::io::savePLYFileASCII(file_name_fused, *(fused_line_cloud.first));
    // visualize_pointclouds_with_orientations(noisy_oriented_pointcloud);
    visualize_pointclouds_with_orientations(fused_line_cloud);

    return 0;
}
