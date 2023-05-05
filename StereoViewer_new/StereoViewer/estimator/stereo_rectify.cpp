#include <iostream>
#include <cmath>

#include <Eigen/Dense>
// #include <opencv2/core/eigen.hpp>

#include "stereo_rectify.h"


void RectifyStereoCamerasByPoints(const Camera& camera,
                                  const std::vector<Eigen::Vector2d>& normal_points1,
                                  const std::vector<Eigen::Vector2d>& normal_points2, 
                                  const std::vector<Eigen::Vector2d>& points1,
                                  const std::vector<Eigen::Vector2d>& points2, 
                                  Eigen::Matrix3d* H1,
                                  Eigen::Matrix3d* H2) {
    Eigen::Matrix3d E = EssentialMatrixEightPointEstimate(normal_points1, normal_points2);
    Eigen::Matrix3d F = camera.intrinsic_matrix.transpose().inverse() * E * camera.intrinsic_matrix.inverse();

    //////////////////// homework ///////////////////////////


    *H1 = Eigen::Matrix3d::Identity();
    *H2 = Eigen::Matrix3d::Identity();
    /////////////////////////////////////////////////////////

}