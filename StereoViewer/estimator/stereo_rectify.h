
#include <Eigen/Core>
#include <Eigen/Dense>

#include "types.h"

#include "camera.h"

#include "essential_matrix.h"

#include <ceres/ceres.h>



// Compute stereo from relative points
void RectifyStereoCamerasByPoints(const Camera& camera,
                                  const std::vector<Eigen::Vector2d>& normal_points1,
                                  const std::vector<Eigen::Vector2d>& normal_points2, 
                                  const std::vector<Eigen::Vector2d>& points1,
                                  const std::vector<Eigen::Vector2d>& points2, 
                                  Eigen::Matrix3d* H1,
                                  Eigen::Matrix3d* H2);

