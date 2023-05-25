#include <iostream>
#include <cmath>

#include <Eigen/Dense>
// #include <opencv2/core/eigen.hpp>

#include "stereo_rectify.h"

Eigen::Vector3d GetE(const std::vector<Eigen::Vector2d>& points,const Eigen::Matrix3d &F)
{
    int length =points.size();
    Eigen::MatrixXd A(length,3);
    // 构建SVD问题
    int i =0;
    for(auto point : points)
    {
        Eigen::Vector3d point_homo=point.homogeneous();
        Eigen::Vector3d l=F.transpose()*point_homo;
        A.block<1,3>(i,0)=l.transpose();
        i++;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d e=svd.matrixV().col(2);
    e=e/e(2);
    return e;
}

Eigen::Matrix3d GetH2(const Eigen::Vector3d &e,const Camera& camera)
{
    //构建T
    Eigen::Matrix3d T;
    // T<<1,0,-camera.intrinsic_matrix(0,2)/2.0,
    //    0,1,-camera.intrinsic_matrix(1,2)/2.0,
    //    0,0,1;
    T<<1.0,0,-double(camera.intrinsic_matrix(0,2)),
       0,1.0,-double(camera.intrinsic_matrix(1,2)),
       0,0,1.0;
    //构建R 这里存疑 是否要用经过T之后的e
    Eigen::Vector3d trans_e=T*e;     
    // std::cout<<"trans_e"<<trans_e<<std::endl;
    Eigen::Matrix3d R;
    double r=sqrt(trans_e(0)*trans_e(0)+trans_e(1)*trans_e(1));
    double alpha=(trans_e(0)>0?1:-1);//alpha 保持了原有的符号
    R<<trans_e(0)/r*alpha,trans_e(1)/r*alpha,0,
       -trans_e(1)/r*alpha,trans_e(0)/r*alpha,0,
       0,0,1;
    Eigen::Vector3d rotated_trans_e=R*trans_e;
    // std::cout<<"rotated trans_e"<<rotated_trans_e<<std::endl;
    //构建G
    Eigen::Matrix3d G;
    G<<1,0,0,0,1,0,-1.0/rotated_trans_e(0),0,1;
//     std::cout<<"G*rotated trans_e"<<G*rotated_trans_e<<std::endl;

//  std::cout<<"T-1*G*rotated trans_e"<<T.inverse()*G*R*T*e<<std::endl;

    
    return T.inverse()*G*R*T;
    // return G*R*T;

}

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
    // 计算极点
    Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
    
    e1 = GetE(normal_points2,F);
    e2 = GetE(normal_points1,F.transpose());
    *H1 = Eigen::Matrix3d::Identity();
    *H2 = Eigen::Matrix3d::Identity();

    // *H1=GetH(e1,camera);
    // std::cout<<"e1':"<<*H1*e1<<std::endl;
    *H2=GetH2(e2,camera);
    // std::cout<<*H2/(*H2).norm()<<std::endl;
    // 计算M
    Eigen::Matrix3d M;
    Eigen::Matrix3d e2x;
    e2x<<0,-e2(2),e2(1),
         e2(2),0,-e2(0),
         -e2(1),e2(0),0;
    M=e2x*F+e2*Eigen::Vector3d(1,1,1).transpose();
    // M=e2x*F;

    
    // Eigen::Matrix3d e2xM=e2.transpose()*M-M.transpose()*e2;
    // M=e2xM/(e2xM.norm())*(F.norm());
    Eigen::MatrixXd A(points1.size(),3);
    Eigen::VectorXd b(points1.size());
    for (int i =0;i<points1.size();++i)
    {
        // 这里把points2写成1了
        Eigen::Vector3d H2x=(*H2)*points2[i].homogeneous();
        // 这里忘记归一化了
        H2x=H2x/H2x(2);
        Eigen::Vector3d H2Mx1=(*H2)*M*points1[i].homogeneous();
        H2Mx1=H2Mx1/H2Mx1(2);
        // std::cout<<"H2x:"<<H2x<<std::endl;
        A.block<1,3>(i,0)=Eigen::Matrix<double,1,3>(H2Mx1(0),H2Mx1(1),1);
        b(i)=-H2x(0);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U=svd.matrixU();
    Eigen::VectorXd b_=U.transpose()*b;
    Eigen::VectorXd s=svd.singularValues();
    for(int i=0;i<b_.size();++i)
    {
        b_(i)=b_(i)/s(i);
    }

    Eigen::Vector3d h1=svd.matrixV()*b_;
    // std::cout<<"h1:"<<h1<<std::endl;
    Eigen::Matrix3d H1_temp;

    H1_temp<<h1(0),h1(1),h1(2),
             0,1,0,
             0,0,1;
    *H1=H1_temp*(*H2)*M;
    // std::cout<<"e2':"<<(*H2)*e2<<std::endl;
    /////////////////////////////////////////////////////////

}