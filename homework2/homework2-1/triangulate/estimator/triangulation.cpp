#include "triangulation.h"
#include <iostream>
Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2) {
  // homework1
  Eigen::Matrix<double,4,4> A;
  
  A.block<1,4>(0,0)=proj_matrix1.row(2)*point1[0]-proj_matrix1.row(0);
  
  A.block<1,4>(1,0)=proj_matrix1.row(2)*point1[1]-proj_matrix1.row(1);
  
  A.block<1,4>(2,0)=proj_matrix2.row(2)*point2[0]-proj_matrix2.row(0);
  
  A.block<1,4>(3,0)=proj_matrix2.row(2)*point2[1]-proj_matrix2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix<double,4,4>> svd(A,Eigen::ComputeFullV);
  Eigen::Matrix4d V=svd.matrixV();
  auto result=V.col(3);
  Eigen::Vector3d point(result[0]/result[3],result[1]/result[3],result[2]/result[3]);
  std::cout<<point<<std::endl;
  return point;

}

std::vector<Eigen::Vector3d> TriangulatePoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  // homework2

  std::vector<Eigen::Vector3d> points3D(points1.size());
    for(int i=0;i<points1.size();++i)
  {
    Eigen::Vector3d Point3D=TriangulatePoint(proj_matrix1,proj_matrix2,points1[i],points2[i]);
    points3D.push_back(Point3D);
  }
  return points3D;
}
