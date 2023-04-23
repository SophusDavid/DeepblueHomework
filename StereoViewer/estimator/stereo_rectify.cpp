#include <iostream>
#include <cmath>

// #include <opencv2/core/eigen.hpp>

#include "stereo_rectify.h"

void RectifyStereoCamerasByPoints(const Camera& camera,
                                  const std::vector<Eigen::Vector2d>& normal_points1,
                                  const std::vector<Eigen::Vector2d>& normal_points2, 
                                  const std::vector<Eigen::Vector2d>& points1,
                                  const std::vector<Eigen::Vector2d>& points2, 
                                  Eigen::Matrix3d* H1,
                                  Eigen::Matrix3d* H2) {
    // homework1：通过匹配的点进行双目立体视觉矫正的H矩阵计算

    
}