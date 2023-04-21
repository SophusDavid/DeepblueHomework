#include <iostream>
#include <vector>
#include <cmath>
#include <float.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

class SingleCamera {
public:
    SingleCamera(Eigen::MatrixXf world_coor, Eigen::MatrixXf pixel_coor, int n)
        : world_coor(world_coor), pixel_coor(pixel_coor), point_num(n),
          P(Eigen::MatrixXf::Zero(2*n, 12)), M(Eigen::MatrixXf::Zero(3, 4)),
          A(Eigen::MatrixXf::Zero(3, 3)), b(Eigen::MatrixXf::Zero(3, 1)),
          K(Eigen::MatrixXf::Zero(3, 3)), R(Eigen::MatrixXf::Zero(3, 3)),
          t(Eigen::MatrixXf::Zero(3, 1)) {}

    void composeP();
    void svdP();
    void workIntrinsicAndExtrinsic();
    void selfcheck(const Eigen::MatrixXf& w_check, const Eigen::MatrixXf& c_check);

private:
    Eigen::MatrixXf world_coor;
    Eigen::MatrixXf pixel_coor;
    int point_num;

    // 变量都是与课程PPT相对应的
    Eigen::MatrixXf P;
    Eigen::MatrixXf M;
    Eigen::MatrixXf A;
    Eigen::MatrixXf b;
    Eigen::MatrixXf K;
    Eigen::MatrixXf R;
    Eigen::MatrixXf t;
};

void SingleCamera::composeP() {
    // homework1: 根据输入的二维点和三维点，构造P矩阵
    
    for (int i = 0; i < world_coor.rows(); i++)
    {
        auto Pi=world_coor.row(i);
        auto ui=pixel_coor.row(i)[0];
        // std::cout<<ui<<std::endl;
        auto vi=pixel_coor.row(i)[1];
        Eigen::Matrix<float,1,12> rowi;
        rowi<<Pi,Eigen::MatrixXf::Zero(1,4),-ui*Pi;
        P.block<1,12>(2*i,0)=rowi;


        Eigen::Matrix<float,1,12> rowi2;
        rowi2<<Eigen::MatrixXf::Zero(1,4),Pi,-vi*Pi;
        // std::cout<<P.rows()<<std::endl;
        P.block<1,12>(2*i+1,0)=rowi2;

    }



    
}

void SingleCamera::svdP() {
    // homework2: 根据P矩阵求解M矩阵和A、b矩阵

    // Eigen::Matrix<float,12,1> b=Eigen::Matrix<float,12,1>::Zero();
    // b(11,0)=1;
    // Eigen::JacobiSVD<Eigen::MatrixXf> svd(P, Eigen::ComputeFullU | Eigen::ComputeFullV);//对矩阵A分解
	// Eigen::Matrix<float,12,12> vt = svd.matrixV().transpose();
    // std::cout<<svd.singularValues()<<std::endl;
    // std::cout<<svd.matrixV()<<std::endl;
    // Eigen::VectorXf x = vt.fullPivHouseholderQr().solve(b);
    // std::cout<<x.transpose()<<std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(P,Eigen::ComputeThinV);
    Eigen::MatrixXf x=svd.matrixV().col(11);
    // Eigen::Matrix<float,3,4> M;
    M.block<1,4>(0,0)=x.block<4,1>(0,0).transpose();
    M.block<1,4>(1,0)=x.block<4,1>(4,0).transpose();
    M.block<1,4>(2,0)=x.block<4,1>(8,0).transpose();
    // std::cout<<P*V.col(9)<<std::endl;

}

void SingleCamera::workIntrinsicAndExtrinsic() {
    // homework3: 求解相机的内参和外参
    A=M.block<3,3>(0,0);
    
    b=M.block<3,1>(0,3);
// std::cout<<M<<std::endl;
    
//     std::cout<<b<<std::endl;

    Eigen::Vector3f a1=A.row(0);
    Eigen::Vector3f a2=A.row(1);
    Eigen::Vector3f a3=A.row(2);
    auto rou =1/a3.norm();
    auto cx=a3.dot(a1)*rou*rou;
    auto cy=a3.dot(a2)*rou*rou;
    // std::cout<<rou*M<<std::endl;
    Eigen::Vector3f a1xa3=a1.cross(a3);
    Eigen::Vector3f a2xa3=a2.cross(a3);
    // std::cout<<a1xa3.dot(a2xa3)<<std::endl;
    auto cos_theta=-(a1xa3.dot(a2xa3))/(a1xa3.norm()*a2xa3.norm());
    auto sin_theta=sqrt(1-cos_theta*cos_theta);
    auto alpha=sin_theta*rou*rou*a1xa3.norm();
    auto beta=sin_theta*rou*rou*a2xa3.norm();
    K(0,0)=alpha;
    K(0,1)=-alpha*cos_theta/sin_theta;
    K(0,2)=cx;
    K(1,1)=beta/sin_theta;
    K(1,2)=cy;
    K(2,2)=1.0;

    auto r3=rou*a3;
    auto r2=(rou*a2-cy*r3)*sin_theta/beta;
    auto r1=(rou*a1-cx*r3+alpha*cos_theta/sin_theta*r2)/alpha;

    R.row(0)=r1;
    R.row(1)=r2;
    R.row(2)=r3;
    t=rou*K.inverse()*b;
    


    std::cout << "K is " <<std::endl<<K<<std::endl;
    std::cout << "R is " <<std::endl<<R<<std::endl;
    std::cout << "t is " <<std::endl<<t.transpose()<<std::endl;
}

void SingleCamera::selfcheck(const Eigen::MatrixXf& w_check, const Eigen::MatrixXf& c_check) {
    float average_err = DBL_MAX;
    // homework4: 根据homework3求解得到的相机的参数，使用测试点进行验证，计算误差
    float error=0;
    Eigen::Matrix<float,3,4> H=Eigen::Matrix<float,3,4>::Zero();
    H.block<3,3>(0,0)=R;
    H.block<3,1>(0,3)=t;
    // H(2,3)=1;
    // std::cout<<H<<std::endl;

    // std::cout<<K<<std::endl;
    for (int i = 0; i < w_check.rows(); i++)
    {
        // std::cout<<H<<std::endl;
        Eigen::Vector4f Pi=w_check.row(i);
        // auto estPnt=M*Pi;
        auto estPnt=K*H*Pi;
        // std::cout<<estPnt<<std::endl;

        
        Eigen::Vector2f res=Eigen::Vector2f(estPnt[0]/estPnt[2],estPnt[1]/estPnt[2])-Eigen::Vector2f(c_check.row(i));
        error+=res.norm();
    }
    average_err=error/w_check.rows();
    
    std::cout << "The average error is " << average_err << "," << std::endl;
    if (average_err > 0.1) {
        std::cout << "which is more than 0.1" << std::endl;
    } else {
        std::cout << "which is smaller than 0.1, the M is acceptable" << std::endl;
    }
}


int main(int argc, char ** argv) {
   
    Eigen::MatrixXf w_xz(4, 4);
    w_xz << 8, 0, 9, 1,
            8, 0, 1, 1,
            6, 0, 1, 1,
            6, 0, 9, 1;

    Eigen::MatrixXf w_xy(4, 4);
    w_xy << 5, 1, 0, 1,
            5, 9, 0, 1,
            4, 9, 0, 1,
            4, 1, 0, 1;

    Eigen::MatrixXf w_yz(4, 4);
    w_yz << 0, 4, 7, 1,
            0, 4, 3, 1,
            0, 8, 3, 1,
            0, 8, 7, 1;

    Eigen::MatrixXf w_coor(12, 4);
    w_coor << w_xz,
            w_xy,
            w_yz;
    // std::cout<<w_coor<<std::endl;
    Eigen::MatrixXf c_xz(4, 2);
    c_xz << 275, 142,
            312, 454,
            382, 436,
            357, 134;

    Eigen::MatrixXf c_xy(4, 2);
    c_xy << 432, 473,
            612, 623,
            647, 606,
            464, 465;

    Eigen::MatrixXf c_yz(4, 2);
    c_yz << 654, 216,
            644, 368,
            761, 420,
            781, 246;

    Eigen::MatrixXf c_coor(12, 2);
    c_coor << c_xz,
            c_xy,
            c_yz;

    Eigen::MatrixXf w_check(5, 4);
    w_check << 6, 0, 5, 1,
                3, 3, 0, 1,
                0, 4, 0, 1,
                0, 4, 4, 1,
                0, 0, 7, 1;

    Eigen::MatrixXf c_check(5, 2);
    c_check << 369, 297,
                531, 484,
                640, 468,
                646, 333,
                556, 194;

    SingleCamera aCamera = SingleCamera(w_coor, c_coor, 12);  // 12 points in total are used
    aCamera.composeP();
    aCamera.svdP();
    aCamera.workIntrinsicAndExtrinsic();
    aCamera.selfcheck(w_check,c_check);  // test 5 points and verify M


    return 0;
}