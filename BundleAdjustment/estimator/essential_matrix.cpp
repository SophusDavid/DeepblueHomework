#include "essential_matrix.h"

#include <complex>
#include "triangulation.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "utils.h"


Eigen::Matrix3d EssentialMatrixEightPointEstimate(const std::vector<Eigen::Vector2d>& points1,
                                                  const std::vector<Eigen::Vector2d>& points2) {

    // Center and normalize image points for better numerical stability.
    std::vector<Eigen::Vector2d> normed_points1;
    std::vector<Eigen::Vector2d> normed_points2;
    Eigen::Matrix3d points1_norm_matrix;
    Eigen::Matrix3d points2_norm_matrix;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

    // Setup homogeneous linear equation as x2' * F * x1 = 0.
    Eigen::Matrix<double, Eigen::Dynamic, 9> cmatrix(points1.size(), 9);
    for (size_t i = 0; i < points1.size(); ++i) {
      cmatrix.block<1, 3>(i, 0) = normed_points1[i].homogeneous();
      cmatrix.block<1, 3>(i, 0) *= normed_points2[i].x();
      cmatrix.block<1, 3>(i, 3) = normed_points1[i].homogeneous();
      cmatrix.block<1, 3>(i, 3) *= normed_points2[i].y();
      cmatrix.block<1, 3>(i, 6) = normed_points1[i].homogeneous();
    }

    // Solve for the nullspace of the constraint matrix.
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> cmatrix_svd(
        cmatrix, Eigen::ComputeFullV);
    const Eigen::VectorXd ematrix_nullspace = cmatrix_svd.matrixV().col(8);
    const Eigen::Map<const Eigen::Matrix3d> ematrix_t(ematrix_nullspace.data());

    // De-normalize to image points.
    const Eigen::Matrix3d E_raw = points2_norm_matrix.transpose() *
                                  ematrix_t.transpose() * points1_norm_matrix;

    // Enforcing the internal constraint that two singular values must be equal
    // and one must be zero.
    Eigen::JacobiSVD<Eigen::Matrix3d> E_raw_svd(
        E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = E_raw_svd.singularValues();
    singular_values(0) = (singular_values(0) + singular_values(1)) / 2.0;
    singular_values(1) = singular_values(0);
    singular_values(2) = 0.0;
    const Eigen::Matrix3d E = E_raw_svd.matrixU() * singular_values.asDiagonal() *
                              E_raw_svd.matrixV().transpose();

    return E;
}

Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
    return CrossProductMatrix(t.normalized()) * R;
}

void DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2, Eigen::Vector3d* t)
{
  //////////////////// homework1 ////////////////
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  W(0, 1) = -1;
  W(1, 0) = 1;
  W(2, 2) = 1;
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  // 保持旋转矩阵行列式为1

  if (U.determinant() < 0)
    U *= -1;
  if (V.determinant() < 0)
    V *= -1;
  Eigen::Matrix3d R1_ = U * W * V.transpose();
  R1_=double(R1_.determinant())*R1_;
  Eigen::Matrix3d R2_ = U * W.transpose() * V.transpose();
  R2_=double(R2_.determinant())*R2_;
  Eigen::Vector3d t1 = U.col(2)*1.0;
  *R1 = R1_;
  *R2 = R2_;
  *t = t1;
  
  ///////////////////////////////////////////////
}

// 上个版本,没用提供的API,自己写 有点问题 还没解决
// void PoseFromEssentialMatrix(const Eigen::Matrix3d& E, const Eigen::Matrix3d &K,
//                              const std::vector<Eigen::Vector2d>& points1,
//                              const std::vector<Eigen::Vector2d>& points2,
//                              Eigen::Matrix3d* R, Eigen::Vector3d* t,
//                              std::vector<Eigen::Vector3d>* points3D) {

//   Eigen::Matrix3d R1;
//   Eigen::Matrix3d R2;
//   DecomposeEssentialMatrix(E, &R1, &R2, t);

//   //////////////////// homework1 ////////////////

//   // 根据DecomposeEssentialMatrix求出的R和t，还有图像中的points1和points2，求解两张图像的相对位姿
//   // 可以参考opencv的modules/calib3d/src/five-point.cpp的recoverPose函数
//   // 此时的到的R有两种情况，t也有两种情况，需要分情况讨论。
//   Eigen::Matrix3x4d P0 = Eigen::Matrix3x4d::Zero();
//   Eigen::Matrix3x4d P1 = Eigen::Matrix3x4d::Zero();
//   Eigen::Matrix3x4d P2 = Eigen::Matrix3x4d::Zero();
//   Eigen::Matrix3x4d P3 = Eigen::Matrix3x4d::Zero();
//   Eigen::Matrix3x4d P4 = Eigen::Matrix3x4d::Zero();
//   P0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
//   P1.block<3, 3>(0, 0) = R1;
//   P1.block<3, 1>(0, 3) = *t;
//   P2.block<3, 3>(0, 0) = R1;
//   P2.block<3, 1>(0, 3) = -(*t);
//   P3.block<3, 3>(0, 0) = R2;
//   P3.block<3, 1>(0, 3) = *t;
//   P4.block<3, 3>(0, 0) = R2;
//   P4.block<3, 1>(0, 3) = -(*t);
  
//   // 构建投影点矩阵 发现triangulation里实现了三角化 直接调用
//   // Eigen::Matrix<double, Eigen::Dynamic, 4> A(points1.size(), 4);
//   // for (size_t i = 0; i < points1.size(); ++i) {
//   //   Eigen::Vector3d p1 = K.inverse() * points1[i].homogeneous();
//   //   Eigen::Vector3d p2 = K.inverse() * points2[i].homogeneous();
//   //   A.block<1, 4>(i, 0) = p1(0) * P1.row(2) - P1.row(0);
//   //   A.block<1, 4>(i, 0) = p1(1) * P1.row(2) - P1.row(1);
//   //   A.block<1, 4>(i, 0) = p2(0) * P2.row(2) - P2.row(0);
//   //   A.block<1, 4>(i, 0) = p2(1) * P2.row(2) - P2.row(1);
//   // }
//   std::vector<Eigen::Vector3d> points3D1=TriangulatePoints( K*P0, K*P1,points1, points2);
//   std::vector<Eigen::Vector3d> points3D2=TriangulatePoints( K*P0, K*P2,points1, points2);
//   std::vector<Eigen::Vector3d> points3D3=TriangulatePoints( K*P0, K*P3,points1, points2);
//   std::vector<Eigen::Vector3d> points3D4=TriangulatePoints( K*P0, K*P4,points1, points2);

//   // 选择正确的解
//   int num1 = 0;
//   int num2 = 0;
//   int num3 = 0;
//   int num4 = 0;
//   for (size_t i = 0; i < points3D1.size(); ++i) {
//     if (points3D1[i](2) > 0)
//       num1++;
//   }
//   for (size_t i = 0; i < points3D2.size(); ++i) {
//     if (points3D2[i](2) > 0)
//       num2++;
//   }
//   for (size_t i = 0; i < points3D3.size(); ++i) {
//     if (points3D3[i](2) > 0)
//       num3++;
//   }
//   for (size_t i = 0; i < points3D4.size(); ++i) {
//     if (points3D4[i](2) > 0)
//       num4++;
//   }

//   if (num1 > num2 && num1 > num3 && num1 > num4) {
//     *R = R1;
//     *t = *t;
//     *points3D = points3D1;
//   } else if (num2 > num1 && num2 > num3 && num2 > num4) {
//     *R = R1;
//     *t = -(*t);
//     *points3D = points3D2;
//   } else if (num3 > num1 && num3 > num2 && num3 > num4) {
//     *R = R2;
//     *t = *t;
//     *points3D = points3D3;
//   } else {
//     *R = R2;
//     *t = -(*t);
//     *points3D = points3D4;
//   }

//   ///////////////////////////////////////////////
// }

void PoseFromEssentialMatrix(const Eigen::Matrix3d& E, const Eigen::Matrix3d &K,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Eigen::Matrix3d* R, Eigen::Vector3d* t,
                             std::vector<Eigen::Vector3d>* points3D) {

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  DecomposeEssentialMatrix(E, &R1, &R2, t);

  //////////////////// homework1 ////////////////
  //求K的逆,将图像坐标转换为相机坐标
  const Eigen::Matrix3d K_inv=K.inverse();
  std::vector<Eigen::Vector2d> points1_cam;
  std::vector<Eigen::Vector2d> points2_cam;
  for (int i =0;i<points1.size();++i)
  {
    points1_cam.push_back((K_inv*points1[i].homogeneous()).hnormalized());
    points2_cam.push_back((K_inv*points2[i].homogeneous()).hnormalized());
  }
  // 存储四个解的R,t 深度为正的点的个数
  std::vector<Eigen::Matrix3d> Rs;
  std::vector<Eigen::Vector3d> ts;
  std::vector<int> depths;
  // 4种情况
  Rs.push_back(R1);
  ts.push_back(*t);
  depths.push_back(0);
  Rs.push_back(R1);
  ts.push_back(-(*t));
  depths.push_back(0);
  Rs.push_back(R2);
  ts.push_back(*t);
  depths.push_back(0);
  Rs.push_back(R2);
  ts.push_back(-(*t));
  depths.push_back(0);
  // 选择正确的解
  for (int i =0;i<4;++i)
  {
    CheckCheirality(Rs[i],ts[i],points1_cam,points2_cam,points3D);
    depths[i]=points3D->size();
  }
  // 获取最大元素，再获取最大元素的索引
  int maxElementIndex = std::max_element(depths.begin(),depths.end()) - depths.begin();
  *R=Rs[maxElementIndex];
  *t=ts[maxElementIndex];

  ///////////////////////////////////////////////
}