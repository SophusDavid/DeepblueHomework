#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/calib3f.hpp>
#include <opencv2/features2d.hpp>
#include <eigen3/Eigen/Core>
// #include <Eigen/Geometry>
// #include <Eigen/LU>
#include <eigen3/Eigen/SVD>
using namespace std;
using namespace cv;
#include <eigen3/Eigen/LU>
// RANSAC参数
int RANSAC_ITERATIONS = 10000;
// const int RANSAC_ITERATIONS = 10000;
// const int RANSAC_ITERATIONS = 1000;

// const double RANSAC_THRESHOLD = 1.0;
const double RANSAC_THRESHOLD = 1.0;

// 伪随机数生成器
RNG rng(12345);

// Point2f转换为Vector2f
Eigen::Vector2f Point2f2Vector2f(const Point2f &p)
{
    Eigen::Vector2f v;
    v << p.x, p.y;
    return v;
}

void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2f> &points,
                                   std::vector<Eigen::Vector2f> *normed_points,
                                   Eigen::Matrix3f *matrix)
{
    // Calculate centroid
    Eigen::Vector2f centroid(0, 0);
    for (const Eigen::Vector2f &point : points)
    {
        centroid += point;
    }
    centroid /= points.size();

    // Root mean square error to centroid of all points
    float rms_mean_dist = 0;
    for (const Eigen::Vector2f &point : points)
    {
        rms_mean_dist += (point - centroid).squaredNorm();
    }
    rms_mean_dist = std::sqrt(rms_mean_dist / points.size());

    // Compose normalization matrix
    const float norm_factor = std::sqrt(2.0) / rms_mean_dist;
    *matrix << norm_factor, 0, -norm_factor * centroid(0), 0, norm_factor,
        -norm_factor * centroid(1), 0, 0, 1;

    // Apply normalization matrix
    normed_points->resize(points.size());

    const float M_00 = (*matrix)(0, 0);
    const float M_01 = (*matrix)(0, 1);
    const float M_02 = (*matrix)(0, 2);
    const float M_10 = (*matrix)(1, 0);
    const float M_11 = (*matrix)(1, 1);
    const float M_12 = (*matrix)(1, 2);
    const float M_20 = (*matrix)(2, 0);
    const float M_21 = (*matrix)(2, 1);
    const float M_22 = (*matrix)(2, 2);

    for (size_t i = 0; i < points.size(); ++i)
    {
        const float p_0 = points[i](0);
        const float p_1 = points[i](1);

        const float np_0 = M_00 * p_0 + M_01 * p_1 + M_02;
        const float np_1 = M_10 * p_0 + M_11 * p_1 + M_12;
        const float np_2 = M_20 * p_0 + M_21 * p_1 + M_22;

        const float inv_np_2 = 1.0 / np_2;
        (*normed_points)[i](0) = np_0 * inv_np_2;
        (*normed_points)[i](1) = np_1 * inv_np_2;
    }
}

// 上次作业的8点法计算本质矩阵，效果贼差，浪费我好长时间 
Eigen::Matrix3f EssentialMatrixEightPointEstimate(const std::vector<Eigen::Vector2f> &points1,
                                                  const std::vector<Eigen::Vector2f> &points2)
{

    // Center and normalize image points for better numerical stability.
    std::vector<Eigen::Vector2f> normed_points1;
    std::vector<Eigen::Vector2f> normed_points2;
    Eigen::Matrix3f points1_norm_matrix;
    Eigen::Matrix3f points2_norm_matrix;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

    Eigen::Matrix3f E;
    E.setIdentity();

    // TODO homework4
    assert(normed_points1.size() == normed_points2.size());
    auto len = normed_points1.size();
    Eigen::MatrixXf A(len, 9);
    for (int i = 0; i < normed_points1.size(); ++i)
    {
        A.row(i) << normed_points1[i][0] * normed_points2[i][0], normed_points1[i][1] * normed_points2[i][0], normed_points2[i][0],
            normed_points1[i][0] * normed_points2[i][1], normed_points1[i][1] * normed_points2[i][1], normed_points2[i][1],
            normed_points1[i][0], normed_points1[i][1], 1;
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
    // std::cout<<svd.singularValues()<<std::endl;
    Eigen::MatrixXf V = svd.matrixV();

    Eigen::Matrix<float, 9, 1> x = V.col(V.cols() - 1);
    E(0, 0) = x[0];
    E(0, 1) = x[1];
    E(0, 2) = x[2];
    E(1, 0) = x[3];
    E(1, 1) = x[4];
    E(1, 2) = x[5];
    E(2, 0) = x[6];
    E(2, 1) = x[7];
    E(2, 2) = x[8];
    // set singular value to zero
    Eigen::JacobiSVD<Eigen::Matrix3f> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd_E.matrixU();
    Eigen::Matrix3f V_E = svd_E.matrixV();
    Eigen::Matrix3f D;
    D.setZero();
    D(0, 0) = (svd_E.singularValues())[0];
    D(1, 1) = (svd_E.singularValues())[1];
    D(2, 2) = 0;
    // std::cout<<U*D*V_E.transpose()<<std::endl; result: E
    E = U * D * V_E.transpose();
    // std::cout<<E<<std::endl;
    // std::cout<<E.norm()<<std::endl; result: 1
    E = points2_norm_matrix.transpose() * E * points1_norm_matrix;

    return E;
}
Mat vec2Mat(const vector<Point2f> &pts)
{
    Mat m(pts.size(), 2, CV_32F);
    for (int i = 0; i < pts.size(); ++i)
    {
        m.at<float>(i, 0) = pts[i].x;
        m.at<float>(i, 1) = pts[i].y;
    }
    return m;
}
// RANSAC求解Essential Matrix
Mat findEssentialMatrixRANSAC(const vector<Point2f> &pts1, const vector<Point2f> &pts2, const Mat &K, vector<uchar> &inliers)
{
    int N = min(pts1.size(), pts2.size());
    int max_inliers = 0;
    float e = 1.0;
    Mat best_E;
    vector<uchar> current_inliers(N);
    int sample_count = 0;
// 原来的循环太久了，改了一下
    while (sample_count < RANSAC_ITERATIONS)
    {
        //  homework2: 选择8个匹配点，并完成Essential matrix的计算
        vector<Point2f> pts1_8, pts2_8;
        for (int i = 0; i < 8; ++i)
        {
            int idx = rng.uniform(0, N);
            pts1_8.push_back(pts1[idx]);
            pts2_8.push_back(pts2[idx]);
        }
        // 自己写的 效果不好
        // Eigen::Matrix3f E=EssentialMatrixEightPointEstimate(pts1_8, pts2_8);
        Mat E = cv::findEssentialMat(pts1_8, pts2_8, K);

        // homework2 end

        // homework3: 计算内点数量
        int inlier_count = 0;
        for (int i = 0; i < N; ++i)
        {

            E.convertTo(E, CV_32F);
            Vec3f p1;
            p1 << pts1[i].x, pts1[i].y, 1.0;
            Vec3f p2;
            p2 << pts2[i].x, pts2[i].y, 1.0;
            // 这一步找到了QIANG的问题，不能除norm，否则会结果变差。
            // Mat p2tEp1 = p2.t() * K.inv().t() * E * K.inv() * p1/norm( E * K.inv() * p1);
            Mat p2tEp1 = p2.t() * K.inv().t() * E * K.inv() * p1;
            if (fabs(p2tEp1.at<float>(0, 0)) < RANSAC_THRESHOLD)
            {
                current_inliers[i] = 1;
                inlier_count++;
            }
        }
        float e_ = 1 - float(inlier_count) / float(N);
        if (e > e_)
        {
            e = e_;
            RANSAC_ITERATIONS = log(1 - 0.99) / log(1 - pow(1 - e, 8));
        }
        // homework3 end
        // homework4: 如果当前内点数量大于最大内点数量，则更新最佳Essential Matrix
        if (inlier_count > max_inliers)
        {
            max_inliers = inlier_count;
            best_E = E;
            inliers = current_inliers;
        }
        sample_count++;
        // homework4 end
    }

    // recompute best E with all inliers
    cout << "Essential Matrix before recompute:" << endl
         << best_E << endl;
    vector<Point2f> pts1_inliers, pts2_inliers;
    for (int i = 0; i < N; ++i)
    {
        if (inliers[i])
        {
            pts1_inliers.push_back(pts1[i]);
            pts2_inliers.push_back(pts2[i]);
        }
    }
    best_E = cv::findEssentialMat(pts1_inliers, pts2_inliers, K);
    return best_E;
}

int main()
{
    // 加载图像并提取特征点
    Mat img1 = imread("../1.JPG", IMREAD_GRAYSCALE);
    Mat img2 = imread("../2.JPG", IMREAD_GRAYSCALE);

    cv::resize(img1, img1, Size(img1.cols * 0.5, img1.rows * 0.5));
    cv::resize(img2, img2, Size(img2.cols * 0.5, img2.rows * 0.5));

    if (img1.empty() || img2.empty())
    {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    vector<Point2f> pts1, pts2;

    // homework1: 提取SIFT特征，并进行特征匹配，可以调用OpenCV的函数
    Ptr<Feature2D> sift = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    sift->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
    vector<DMatch> matches;
    // FlannBasedMatcher matcher;
    BFMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);
    // 找到优质匹配点 这个对结果提升较大
    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < descriptors1.rows; ++i)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    // vector<DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; ++i)
    {
    //     if (matches[i].distance <= max(2.5 * min_dist, 30.0))
    //     {
            pts1.push_back(keypoints1[matches[i].queryIdx].pt);
            pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    //         good_matches.push_back(matches[i]);
    //     }
    }
    // homework1 end

    // 假设相机内参矩阵已知
    Mat K = (Mat_<float>(3, 3) << 1278, 0, 768, 0, 1278, 576, 0, 0, 1);

    // 使用RANSAC估计Essential Matrix
    vector<uchar> inliers;
    Mat E = findEssentialMatrixRANSAC(pts1, pts2, K, inliers);

    cout << "Essential Matrix:" << endl
         << E << endl;

    Mat E_cv = findEssentialMat(pts1, pts2, K, RANSAC, 0.99, 1.0);
    cout << "Essential Matrix From OpenCV:" << endl
         << E_cv << endl;

    Mat E_test = findEssentialMat(pts1, pts2, K);
cout << "Essential Matrix test:" << endl
         << E_test << endl;
    // 可视化内点匹配
    vector<DMatch> inlier_matches;

    // 由于特征点匹配过于密集，可视化的效果不够明显，可以将 i++ 改为 i += 20
    for (size_t i = 0; i < inliers.size(); i++)
    {
        if (inliers[i])
        {
            // inlier_matches.push_back(good_matches[i]);
            inlier_matches.push_back(matches[i]);

        }
    }

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, img_matches);
    imshow("Inlier Matches", img_matches);
    imwrite("./inlier_matches.png", img_matches);
    // waitKey();

    return 0;
}
