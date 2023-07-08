#include "LineFusion.hpp"
#include <cmath>
#include <limits>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cassert>
#include <Eigen/Dense>
#include <pcl/common/point_tests.h> // for pcl::isFinite
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

#define assertm(exp, msg) assert(((void)msg, exp))

#define DISTANCE_FUSION 2e-6
#define KD_TREE_RADIUS 0.002
#define CURVE_ANGLE (2 * (M_1_PI * M_1_PI) / 36.)
#define STRAND_THICKNESS (2 * 1e-8)

void LineFusion::line_fusion(OrientedPointCloud &fused_line_cloud)
{
    fused_line_cloud.second.reserve(m_original_points.second.size()); // Reserve areas for direction vector
    EigenLine q_prev, q_next;
    float d;
    assertm(m_original_points.second.size() == m_original_points.first->size(), "Point and direction vector sizes should be equal.");
    for (size_t i = 0; i < m_original_points.second.size(); i++)
    {
        q_prev.first = m_original_points.first->at(i).getArray3fMap().cast<float>(); // x-y-z points
        q_prev.second = m_original_points.second.at(i);                              // direction
        d = DISTANCE_FUSION + 1;                                                     // Just assigned a value bigger than distance fusion
        int threshold = 0;
        while (d > DISTANCE_FUSION && threshold < 1000)
        {
            q_next = local_meanshift(q_prev, i);
            d = (q_next.first - q_prev.first).norm();
            q_prev = q_next;
            threshold++;
        }
        fused_line_cloud.first->emplace_back(q_next.first[0], q_next.first[1], q_next.first[2]);
        fused_line_cloud.second.push_back(q_next.second);
        std::cout << i << std::endl;
    }
}

std::optional<LineFusion::EigenVector> LineFusion::line_plane_intersection(const LineFusion::EigenVector &point_on_plane, const LineFusion::EigenVector &plane_normal, const LineFusion::EigenLine &line)
{
    // If the dot product between the line and the plane normal is zero,
    // then the plane and the line are parallel to each other which means there's no intersection.

    if (abs(line.second.dot(plane_normal)) <= 2 * std::numeric_limits<float>::epsilon())
    {
        return {};
    }
    float numerator = (point_on_plane - line.first).dot(plane_normal.stableNormalized());
    float denominator = line.second.stableNormalized().dot(plane_normal.stableNormalized());
    float d = numerator / denominator;

    return line.first + d * line.second.stableNormalized();
}

LineFusion::EigenLine LineFusion::local_meanshift(const LineFusion::EigenLine &q, const int index)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    kdtree.setInputCloud(m_original_points.first);

    pcl::PointXYZ searchPoint{static_cast<float>(q.first[0]), static_cast<float>(q.first[1]), static_cast<float>(q.first[2])};

    std::vector<int> point_idx_radius_search;
    std::vector<float> point_radius_squared_distance;
    if (pcl::isFinite(searchPoint) && kdtree.radiusSearch(searchPoint, KD_TREE_RADIUS, point_idx_radius_search, point_radius_squared_distance) > 1)
    {
        LineFusion::EigenVector init_position{m_original_points.first->at(index).getArray3fMap()};
        LineFusion::EigenVector init_direction{m_original_points.second[index]};
        LineFusion::EigenVector position, direction;
        std::vector<LineFusion::EigenVector> positions_all, directions_all;
        LineFusion::EigenLine line;
        float weight = 0;
        std::vector<float> weights_all;
        std::optional<LineFusion::EigenVector> new_position;
        float total_weights = 0;
        for (auto &&neighbor : point_idx_radius_search)
        {
            position = m_original_points.first->at(neighbor).getArray3fMap();
            direction << m_original_points.second[neighbor];
            line.first << position;
            line.second << direction;
            new_position = line_plane_intersection(init_position, init_direction, line);
            if (new_position.has_value())
            {
                float squared_diff = (new_position.value() - init_position).squaredNorm();
                float first_part = squared_diff / STRAND_THICKNESS;

                float dir_diff = init_direction.stableNormalized().dot(direction.stableNormalized());
                float second_part = pow(acos(dir_diff), 2) / CURVE_ANGLE;
                weight = exp(-(first_part + second_part));
                positions_all.push_back(new_position.value());
                directions_all.push_back(direction);
                weights_all.push_back(weight);
                total_weights += weight;
            }
        }

        LineFusion::EigenLine mean_line{{0, 0, 0}, {0, 0, 0}};
        if (total_weights > 2 * std::numeric_limits<float>::epsilon())
        {
            for (size_t i = 0; i < positions_all.size(); i++)
            {
                // mean_line.first += weights_all.at(i) * positions_all.at(i);
                // mean_line.second += weights_all.at(i) * directions_all.at(i);
                mean_line.first += positions_all.at(i);
                mean_line.second += directions_all.at(i);
            }
            mean_line.first /= positions_all.size();
            mean_line.second /= positions_all.size();
        }
        else
        {
            mean_line = q;
        }
        return mean_line;
    }
    return q;
}
