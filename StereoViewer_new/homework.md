# 双目立体视觉课程作业  

本次作业使用的框架与Essential Matrix估计课程的框架相同，依旧是为大家提供了小房子模型和相机运动的轨迹。

根据课上老师的讲解，相信大家已经对双目立体视觉有了深入的了解。

本次作业的题目是：

给出大家两张图像上匹配的点，通过匹配的点进行双目立体视觉矫正的H矩阵计算。  
其中主要需要大家实现的函数是RectifyStereoCamerasByPoints，其中，输入包括了归一化之后的points和原始的points，可以用于求Essential矩阵或者Fundamental矩阵，这些参数不一定是需要全部用上的。

目前在RectifyStereoCamerasByPoints中，已经为大家提供了E矩阵和F矩阵的结果。  

为了实现对RectifyStereoCamerasByPoints函数实现的正确性的检查，在main函数中调用了opencv的stereoRectifyUncalibrated函数，这个函数的我们自己实现的函数在理想状态下，要与stereoRectifyUncalibrated函数的输出在归一化之后的值是接近的。如果在实现RectifyStereoCamerasByPoints过程中遇上困难，也可以参考stereoRectifyUncalibrated函数的实现。

通过全局搜索homework，快速定位作业位置

编译运行方式：
    
    编译：  
    cd StereoViewer
    mdkir build
    cd build
    cmake ..
    make -j

    运行：
    ./stereo ../data/house.txt  