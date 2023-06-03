#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "EigenQuaternionParameterization.h"

#include "common_include.h"
#include "frame.h"
#include "map.h"
#include "feature.h"

namespace sfm {

class ReprojectionErrorAutoDiff
{
public:

    ReprojectionErrorAutoDiff(const Eigen::Vector2d& observed_p, 
                              double fx_, double fy_,
                              double cx_, double cy_)
        :  m_observed_p(observed_p),
          fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    // variables: camera extrinsics and point position
    // ReprojectionErrorAutoDiff后面第一个参数2表示残差维度，第二个参数4表示q_c1_w的维度，第三个参数3表示t_c1_w的维度，第四个参数3表示landmark的维度
    template <typename T>
    bool operator()(const T* const q,
                    const T* const t,
                    const T* const P_w,
                    T* residuals) const
    {
        /////////////////////// homework2 ////////////////////////////
        Eigen::Matrix<T,2,1> observed_p_=m_observed_p.cast<T>();
        T  P_c[3];
        // 读取四元数
        T q_ceres[4] = {q[3], q[0], q[1], q[2]};
        // 使用自带的旋转函数 这个地方可以一个数量级的误差 ::WHY?
        ceres::QuaternionRotatePoint(q_ceres,P_w,P_c);
        // 平移
        P_c[0] += t[0];
        P_c[1] += t[1];
        P_c[2] += t[2];
        
        // 读取相机内参
        T fx_ = T(fx);
        T fy_ = T(fy);
        T cx_ = T(cx);
        T cy_ = T(cy);
        // 归一化坐标系下的点
        T p_[2];
        p_[0]=fx_ * P_c[0] / P_c[2] + cx_; 
        p_[1]=fy_ * P_c[1] / P_c[2] + cy_;
        // 误差
        residuals[0] = p_[0]  - observed_p_[0];
        residuals[1] = p_[1]  - observed_p_[1];

        // residuals[0] = T(0);
        // residuals[1] = T(0);

        /////////////////////// homework2 ////////////////////////////

        return true;
    }

     
    // observed 2D point
    Eigen::Vector2d m_observed_p;
    // Camera intrinsic
    double fx, fy, cx, cy;
};

class TwoFrameBundleAdjuster {
public:

    TwoFrameBundleAdjuster(double fx_, double fy_,
                           double cx_, double cy_)
        :fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    void SetLastFrame(Frame::Ptr last_frame) {
        last_frame_ = last_frame;
    }

    void SetCurrentFrame(Frame::Ptr current_frame){
        current_frame_ = current_frame;
    }

    void SetMap(Map::Ptr map) { map_ = map; }

    void Optimize();

private:

    Frame::Ptr last_frame_ = nullptr;
    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;

    std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;

    double fx, fy, cx, cy;
};


class BundleAdjuster {
public:

    BundleAdjuster(double fx_, double fy_,
                   double cx_, double cy_)
        :fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    void SetMap(Map::Ptr map) { map_ = map; }

    void Optimize();

private:

    Frame::Ptr last_frame_ = nullptr;
    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;

    std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;

    double fx, fy, cx, cy;
};


}