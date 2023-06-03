#include "ceres_optim.h"

namespace sfm {

void TwoFrameBundleAdjuster::Optimize() {
    ceres::Problem problem;

    // 原始的Pose应该是 Twc，从相机到世界坐标系的定义
    Eigen::Quaterniond q_c1_w = Eigen::Quaterniond( last_frame_->Pose().inverse().linear() );
    Eigen::Vector3d t_c1_w = Eigen::Vector3d(0.001, 0.001, 0.001);

    Eigen::Quaterniond q_c2_w = Eigen::Quaterniond( current_frame_->Pose().inverse().linear() );
    Eigen::Vector3d t_c2_w = current_frame_->Pose().inverse().translation();

    std::vector<Eigen::Vector3d> points;
    points.clear();
    std::vector<int> points_idx;
    points_idx.clear();

    active_landmarks_ = map_->GetActiveMapPoints();

    for (auto& landmark : active_landmarks_) {
        points_idx.push_back(landmark.first);
        auto pos = landmark.second->Pos();
        points.push_back(pos);
    }

    for(int i = 0; i < points.size(); i++) {

        for(auto& feature : active_landmarks_[points_idx[i]]->observations_){

            auto feat = feature.lock();

            Eigen::Vector2d observed_p(feat->position_.pt.x, feat->position_.pt.y);

            auto frame = feat->frame_.lock();

            if(frame->id_ == 0) {
                // ReprojectionErrorAutoDiff后面第一个参数2表示残差维度，第二个参数4表示q_c1_w的维度，第三个参数3表示t_c1_w的维度，第四个参数3表示landmark的维度
                ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                  new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));
                
                ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);

                problem.AddResidualBlock(costFunction, lossFunction, 
                                         q_c1_w.coeffs().data(), t_c1_w.data(), points.at(i).data());
            } else {
                ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                  new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));
                
                ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);

                problem.AddResidualBlock(costFunction, lossFunction, 
                                         q_c2_w.coeffs().data(), t_c2_w.data(), points.at(i).data());
            }
        }

    }

    ceres::LocalParameterization* quaternionParameterization =
            new EigenQuaternionParameterization;

    problem.SetParameterization(q_c1_w.coeffs().data(), quaternionParameterization);
    problem.SetParameterization(q_c2_w.coeffs().data(), quaternionParameterization);

    // 把第一帧的位姿设置为Identity
    problem.SetParameterBlockConstant(q_c1_w.coeffs().data());
    problem.SetParameterBlockConstant(t_c1_w.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    for(int i = 0; i < points_idx.size(); i++) {
        active_landmarks_[points_idx[i]]->SetBAPos(points[i]);
        active_landmarks_[points_idx[i]]->optimized = true;
    }

    Eigen::Affine3d Tc2w_update;
    Tc2w_update.linear() = q_c2_w.toRotationMatrix();
    Tc2w_update.translation() = t_c2_w;

    current_frame_->SetBAPose(Tc2w_update.inverse());
    current_frame_->optimized = true;

    std::cout << "end optim "<<std::endl;
}

void BundleAdjuster::Optimize() {
    ceres::Problem problem;
    ceres::LocalParameterization* quaternionParameterization =
            new EigenQuaternionParameterization;

    active_keyframes_ = map_ -> GetActiveKeyFrames();
    active_landmarks_ = map_ -> GetActiveMapPoints();

    ////////////////////////// homework3 //////////////////////
    
    std::unordered_map<unsigned long, Eigen::Quaterniond> q_cw;
    std::unordered_map<unsigned long, Eigen::Vector3d> t_cw;
    std::unordered_map<unsigned long, Vec3> points;
    // 存储每一帧的位姿
    for(auto keyframe:active_keyframes_){
        // 获取每一帧的位姿
        Eigen::Quaterniond q_c2_w = Eigen::Quaterniond(keyframe.second->Pose().inverse().linear());
        Eigen::Vector3d t_c2_w = keyframe.second->Pose().inverse().translation();
        q_cw.emplace(keyframe.first , q_c2_w);
        t_cw.emplace(keyframe.first, t_c2_w);
    }
    for (auto& landmark : active_landmarks_) {
        // 获取共视点
        std::list<std::weak_ptr<Feature>> features = landmark.second->GetObs();
        points.emplace(landmark.first, landmark.second->Pos());
        for (auto& feature : features) {
            // 获取共视点所在的帧
            std::shared_ptr<Frame> frame = feature.lock()->frame_.lock();
            // 如果共视点所在的帧不是第一帧，就添加残差项
            // if (frame->id_ != 0) {
                // 获取共视点的像素坐标
                Eigen::Vector2d observed_p(feature.lock()->position_.pt.x, feature.lock()->position_.pt.y);

                            
                // 添加残差项
                ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                  new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));
                ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);
                problem.AddResidualBlock(costFunction, lossFunction, 
                                         q_cw[frame->id_].coeffs().data(), t_cw[frame->id_].data(), points[landmark.first].data());
        }
    }
    
    for (const auto& keyframe: active_keyframes_) {
        auto keyframe_id = keyframe.first;
        auto frame_ptr = keyframe.second;
        auto frame_id = frame_ptr->id_;

        problem.SetParameterization(q_cw[keyframe_id].coeffs().data(), quaternionParameterization);

        // 第一个关键帧的姿态通常被选作参考姿态，不需要进行优化
        if (frame_id == 0) {
            problem.SetParameterBlockConstant(q_cw[keyframe_id].coeffs().data());
            problem.SetParameterBlockConstant(t_cw[keyframe_id].data());
        }
    }
    //配置优化问题
    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;

    // 遍历umap,更新位姿
   // Update 3d points
    for (auto &landmark: active_landmarks_) {
        landmark.second->SetBAPos(points[landmark.first]);
        landmark.second->optimized = true;
    }

    // Update poses
    for (auto &keyframe: active_keyframes_) {
        auto keyframe_id = keyframe.first;

        Eigen::Affine3d T_update;
        T_update.linear() = q_cw[keyframe_id].toRotationMatrix();
        T_update.translation() = t_cw[keyframe_id];

        keyframe.second->SetBAPose(T_update.inverse());
        keyframe.second->optimized = true;
    }

    std::cout << "end optim "<<std::endl;
    
    //////////////////////////// homework3 //////////////////////////
}

}
