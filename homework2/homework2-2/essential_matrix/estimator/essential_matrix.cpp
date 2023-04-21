#include "essential_matrix.h"

#include <complex>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include<iostream>
void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* matrix) {
    // Calculate centroid
    Eigen::Vector2d centroid(0, 0);
    for (const Eigen::Vector2d& point : points) {
        centroid += point;
    }
    centroid /= points.size();

    // Root mean square error to centroid of all points
    double rms_mean_dist = 0;
    for (const Eigen::Vector2d& point : points) {
        rms_mean_dist += (point - centroid).squaredNorm();
    }
    rms_mean_dist = std::sqrt(rms_mean_dist / points.size());

    // Compose normalization matrix
    const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
    *matrix << norm_factor, 0, -norm_factor * centroid(0), 0, norm_factor,
        -norm_factor * centroid(1), 0, 0, 1;

    // Apply normalization matrix
    normed_points->resize(points.size());

    const double M_00 = (*matrix)(0, 0);
    const double M_01 = (*matrix)(0, 1);
    const double M_02 = (*matrix)(0, 2);
    const double M_10 = (*matrix)(1, 0);
    const double M_11 = (*matrix)(1, 1);
    const double M_12 = (*matrix)(1, 2);
    const double M_20 = (*matrix)(2, 0);
    const double M_21 = (*matrix)(2, 1);
    const double M_22 = (*matrix)(2, 2);

    for (size_t i = 0; i < points.size(); ++i) {
        const double p_0 = points[i](0);
        const double p_1 = points[i](1);

        const double np_0 = M_00 * p_0 + M_01 * p_1 + M_02;
        const double np_1 = M_10 * p_0 + M_11 * p_1 + M_12;
        const double np_2 = M_20 * p_0 + M_21 * p_1 + M_22;

        const double inv_np_2 = 1.0 / np_2;
        (*normed_points)[i](0) = np_0 * inv_np_2;
        (*normed_points)[i](1) = np_1 * inv_np_2;
    }
}

Eigen::Matrix3d EssentialMatrixEightPointEstimate(const std::vector<Eigen::Vector2d>& points1,
                                                  const std::vector<Eigen::Vector2d>& points2) {

    // Center and normalize image points for better numerical stability.
    std::vector<Eigen::Vector2d> normed_points1;
    std::vector<Eigen::Vector2d> normed_points2;
    Eigen::Matrix3d points1_norm_matrix;
    Eigen::Matrix3d points2_norm_matrix;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

    Eigen::Matrix3d E;
    E.setIdentity();
    
    // TODO homework4
    assert(normed_points1.size()==normed_points2.size());
    auto len=normed_points1.size();
    Eigen::MatrixXd A(len,9);
    for (int i=0;i<normed_points1.size();++i)
    {
        A.row(i)<<normed_points1[i][0]*normed_points2[i][0],normed_points1[i][1]*normed_points2[i][0],normed_points2[i][0],
        normed_points1[i][0]*normed_points2[i][1],normed_points1[i][1]*normed_points2[i][1],normed_points2[i][1],
        normed_points1[i][0],normed_points1[i][1],1;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeThinV);
    Eigen::Matrix<double,9,1> x=svd.matrixV().col(8);
    E(0,0)=x[0];
    E(0,1)=x[1];
    E(0,2)=x[2];
    E(1,0)=x[3];
    E(1,1)=x[4];
    E(1,2)=x[5];
    E(2,0)=x[6];
    E(2,1)=x[7];
    E(2,2)=x[8];
    // std::cout<<E.norm()<<std::endl; result: 1
    E=points2_norm_matrix.transpose()*E*points1_norm_matrix;
    
    
    return E;
}

Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
    Eigen::Matrix3d E;
    E.setIdentity();

    // TODO homework3
    auto ax=t[0];
    auto ay=t[1];
    auto az=t[2];
    Eigen::Matrix3d tx;
    tx<<0,-az,ay
        ,az,0,-ax
        ,-ay,ax,0;
    E=tx*R;
    
    return E;
}

