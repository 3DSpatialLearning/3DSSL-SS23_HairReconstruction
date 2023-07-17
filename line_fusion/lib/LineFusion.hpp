#pragma once

#include <vector>
#include <array>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <optional>
// TODO: Visualize the result and the end term
class LineFusion
{
public:
    typedef pcl::PointXYZ Point;
    typedef Eigen::Vector3f EigenVector;
    typedef std::pair<EigenVector, EigenVector> EigenLine;
    typedef std::vector<std::pair<EigenVector, EigenVector>> EigenLineCloud;
    typedef std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<EigenVector>> OrientedPointCloud;

    LineFusion(OrientedPointCloud oriented_pointcloud) : m_original_points{oriented_pointcloud} {}
    void line_fusion(OrientedPointCloud &fused_line_cloud);

private:
    std::optional<EigenVector> line_plane_intersection(const EigenVector &point_on_plane, const EigenVector &plane_normal, const EigenLine &line);
    EigenLine local_meanshift(const EigenLine &q, const int index, const pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree);
    const OrientedPointCloud m_original_points;
};