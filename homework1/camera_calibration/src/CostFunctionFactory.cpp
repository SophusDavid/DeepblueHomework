#include "CostFunctionFactory.h"
#include "ceres/ceres.h"
#include "PinholeCamera.h"
#include <Eigen/Core>

class ReprojectionErrorAutoDiff
{
public:
    ReprojectionErrorAutoDiff(const Eigen::Vector3d &observed_P,
                              const Eigen::Vector2d &observed_p)
        : m_observed_P(observed_P), m_observed_p(observed_p) {}

    // variables: camera intrinsics and camera extrinsics
    template <typename T>
    bool operator()(const T *const params, // 8 k1 k2 p1 p2 fx fy cx cy
                    const T *const q,      // 4 四元数
                    const T *const t,      // 3
                    T *residuals) const
    {
         Eigen::Matrix<T, 3, 1> P = m_observed_P.cast<T>();
         Eigen::Matrix<T, 2, 1> predicted_p;

         Eigen::Matrix<T, 2, 1> e = Eigen::Matrix<T, 2, 1>::Zero();
        // TODO: homework2

        // 完成相机的投影过程，计算重投影误差
        //此处有个坑就是四元数导出是x,y,z,w 。初始化是w,x,y,z
    //    const  Eigen::Quaternion<T> quaternion(q[3],q[0], q[1], q[2] );
    //     const   Eigen::Matrix<T, 3, 3> rotation_matrix = quaternion.toRotationMatrix();
    //    const  Eigen::Matrix<T, 3, 1> trans_vec(t[0], t[1], t[2]);
    //     // Eigen::Matrix<T, 3, 3> K;
    //     // K << params[4], T(0.0), params[6],
    //     //     T(0.0), params[5], params[7],
    //     //     T(0), T(0), T(1);
    //     Eigen::Matrix<T, 3, 1> prejected_p_dist = rotation_matrix * P +trans_vec;
    //     // Eigen::Matrix<T, 3, 1> prejected_p_dist = K*(rotation_matrix * P +trans_vec);


    //     // T P_w[3];
    //     // P_w[0] = T(P(0));
    //     // P_w[1] = T(P(1));
    //     // P_w[2] = T(P(2));
    //     // T q_ceres[4] = {q[3], q[0], q[1], q[2]};

    //     // T P_c[3];
    //     // ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

    //     // P_c[0] += t[0];
    //     // P_c[1] += t[1];
    //     // P_c[2] += t[2];

    //     T x = prejected_p_dist[0] / prejected_p_dist[2];
    //     T y = prejected_p_dist[1] / prejected_p_dist[2];
    //             // T x = P_c[0] / P_c[2];
    //     // T y = P_c[1] / P_c[2];
    //     // Eigen::Matrix<T,2,1> estPoint(x,y);

    //     T k1 = params[0];
    //     T k2 = params[1];
    //     T p1 = params[2];
    //     T p2 = params[3];

    //     T mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
    //     mx2_u = x * x;
    //     my2_u = y * y;
    //     mxy_u = x * y;
    //     rho2_u = mx2_u + my2_u;
    //     rad_dist_u =1.0+ k1 * rho2_u + k2 * rho2_u * rho2_u;


    //     T x_ = x * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);

    //     T y_ = y * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    //     x_ = (x_ ) * params[4] + params[6];
    //     y_ = (y_ ) * params[5] + params[7];
    //     // T x_ = ( x) * params[4] + params[6];
    //     // T y_ = ( y) * params[5] + params[7];
    //     residuals[0] =  m_observed_p[0]-x_;
    //     residuals[1] =  m_observed_p[1]-y_;
        // Eigen::Matrix<T,2,1> m_observed_p_=m_observed_p.cast<T>();
        // e=m_observed_p_-estPoint;
        // residuals[0] =e(0);
        // residuals[1] =e(1);

        T P_w[3];
        P_w[0] = T(P(0));
        P_w[1] = T(P(1));
        P_w[2] = T(P(2));

        // Convert quaternion from Eigen convention (x, y, z, w)
        // to Ceres convention (w, x, y, z)
        T q_ceres[4] = {q[3], q[0], q[1], q[2]};

        T P_c[3];
        ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

        P_c[0] += t[0];
        P_c[1] += t[1];
        P_c[2] += t[2];

        // project 3D object point to the image plane
        T k1 = params[0];
        T k2 = params[1];
        T p1 = params[2];
        T p2 = params[3];
        T fx = params[4];
        T fy = params[5];
        T cx = params[6];
        T cy = params[7];

        // Transform to model plane
        T u = P_c[0] / P_c[2];
        T v = P_c[1] / P_c[2];

        T rho_sqr = u * u + v * v;
        T L = T(1.0) + k1 * rho_sqr + k2 * rho_sqr * rho_sqr;
        T du = T(2.0) * p1 * u * v + p2 * (rho_sqr + T(2.0) * u * u);
        T dv = p1 * (rho_sqr + T(2.0) * v * v) + T(2.0) * p2 * u * v;

        u = L * u + du;
        v = L * v + dv;
        predicted_p(0) = fx * u + cx;
        predicted_p(1) = fy * v + cy;
        e = predicted_p - m_observed_p.cast<T>();
        residuals[0] = e(0);
        residuals[1] = e(1);

        return true;
    }

    // observed 3D point
    Eigen::Vector3d m_observed_P;

    // observed 2D point
    Eigen::Vector2d m_observed_p;
};

boost::shared_ptr<CostFunctionFactory> CostFunctionFactory::m_instance;

CostFunctionFactory::CostFunctionFactory()
{
}

boost::shared_ptr<CostFunctionFactory>
CostFunctionFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CostFunctionFactory);
    }

    return m_instance;
}

ceres::CostFunction *
CostFunctionFactory::generateCostFunction(const PinholeCameraConstPtr &camera,
                                          const Eigen::Vector3d &observed_P,
                                          const Eigen::Vector2d &observed_p) const
{

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);
    ceres::CostFunction *costFunction = nullptr;

    costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 8, 4, 3>(
        new ReprojectionErrorAutoDiff(observed_P, observed_p));

    return costFunction;
}
