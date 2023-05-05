#include "camera.h"
#include "visualizer.h"
#include "essential_matrix.h"
#include "stereo_rectify.h"
#include "utils.h"

#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/eigen.hpp>


int main(int argc, char **argv){
    Camera camera = Camera(std::string(argv[1]));
    camera.GenerateFrames();

    Eigen::Matrix4d Twc1 = camera.GetFirstFrameTwc();
    Eigen::Matrix4d Twc2 = camera.GetSecondFrameTwc();

    std::vector<Eigen::Vector2d> points1 = camera.GetFirstFramePoints();
    std::vector<Eigen::Vector2d> points2 = camera.GetSecondFramePoints();

    std::vector<Eigen::Vector2d> normal_points1 = camera.GetFirstFrameNormalPoints();
    std::vector<Eigen::Vector2d> normal_points2 = camera.GetSecondFrameNormalPoints();

    Eigen::Matrix3d H1, H2;

    Eigen::Matrix4d Tc2c1 = Twc2.inverse() * Twc1;
    
    Eigen::Quaterniond Qc2c1 = Eigen::Quaterniond(Tc2c1.block<3, 3>(0, 0));
    Eigen::Vector3d tc2c1 = Tc2c1.block<3, 1>(0, 3);

    RectifyStereoCamerasByPoints(camera, normal_points1, normal_points2, points1, points2, &H1, &H2);

    double H1_norm = H1.norm();
    double H2_norm = H2.norm();
    H1 = H1 / H1_norm;
    H2 = H2 / H2_norm;
    std::cout << "H1" <<std::endl;
    std::cout << H1 <<std::endl;

    std::cout << "H2" <<std::endl;
    std::cout << H2 <<std::endl;

    cv::Mat out1(camera.image_h,camera.image_w, CV_8UC3, cv::Scalar(255,255,255));
    cv::Mat homo1_;
    cv::eigen2cv(H1, homo1_);

    cv::Mat out2(camera.image_h,camera.image_w, CV_8UC3, cv::Scalar(255,255,255));
    cv::Mat homo2_;
    cv::eigen2cv(H2, homo2_);

    cv::warpPerspective(camera.GetFirstFrameImage(), out1, homo1_, cv::Size(640, 480));
    cv::warpPerspective(camera.GetSecondFrameImage(), out2, homo2_, cv::Size(640, 480));

    std::vector<cv::Point2f> points1_cv = convertPointsOfEigenToCV(points1);
    std::vector<cv::Point2f> points2_cv = convertPointsOfEigenToCV(points2);

    cv::Mat F_cv = cv::findFundamentalMat(cv::Mat(points1_cv), cv::Mat(points2_cv));
    cv::Mat H_cv1, H_cv2;
    cv::stereoRectifyUncalibrated(cv::Mat(points1_cv), cv::Mat(points2_cv), F_cv,
                                 cv::Size(camera.image_w, camera.image_h), H_cv1, H_cv2, 0.99);

    std::vector<cv::Point2f> points1_trans;
    std::vector<cv::Point2f> points2_trans;

    cv::perspectiveTransform(points1_cv, points1_trans, homo1_);
    cv::perspectiveTransform(points2_cv, points2_trans, homo2_);

    for(int i = 0; i < points1_trans.size(); i++) {
        cv::circle(out1, cv::Point(int(points1_trans[i].x), int(points1_trans[i].y)), 3, cv::Scalar(255, 0, 0));
        cv::circle(out2, cv::Point(int(points2_trans[i].x), int(points2_trans[i].y)), 3, cv::Scalar(255, 0, 0));
    }

    cv::imshow("warp1", out1);
    cv::waitKey(0);

    cv::imshow("warp2", out2);
    cv::waitKey(0);

    cv::cv2eigen(H_cv1, H1);
    cv::cv2eigen(H_cv2, H2);

    H1_norm = H1.norm();
    H2_norm = H2.norm();
    H1 = H1 / H1_norm;
    H2 = H2 / H2_norm;
    std::cout << "H1" <<std::endl;
    std::cout << H1 <<std::endl;

    std::cout << "H2" <<std::endl;
    std::cout << H2 <<std::endl;

    return 0;
}