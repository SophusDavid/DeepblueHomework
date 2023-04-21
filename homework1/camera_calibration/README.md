# 相机标定算法

- 依赖的三方库：
1. Eigen3
2. ceres2.0版本
3. OpenCV 4.x
4. yaml-cpp

- 编译方式：
mkdir build 
cd build
cmake ..
make -j4

- 使用方式：
./camera_calib -i 图像数据集的地址 

- 如果使用基于apriltag的标定板
./camera_calib -i 图像数据集的地址 -a true

- 使用的数据集：
Euroc的标定数据集 [chessboard](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/calibration_datasets/cam_checkerboard/cam_checkerboard.zip) [apriltag](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/calibration_datasets/cam_april/cam_april.zip)

或者是百度云链接：https://pan.baidu.com/s/1wJojGAZP8wsAGywb4Rn4VQ?pwd=2vgd 

- tips: 图像数据集的地址 需要将地址定义到最后一层 cam_april/mav0/cam0/data

作业相关的docker：https://hub.docker.com/repository/docker/xxxxxl/reconstruction/general 
 
使用这个docker需要安装：docker 和 nvidia-container-toolkit