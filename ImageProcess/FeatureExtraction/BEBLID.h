#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>												//标准C++库中的输入输出类相关头文件。
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include <utility>
#include <chrono>
class BEBLID : public cv::Feature2D
{
public:
    /** @brief Creates the BEBLID descriptor.
    //创建BEBLID描述符。
    @param n_wls The number of final weak-learners in the descriptor. It must be a multiple of 8 such as 256 or 512.
    //描述符中最终弱学习器的数量。必须是8的整数倍，例如256或512。
    @param scale_factor Adjusts the sampling window of detected keypoints
    //scale_factor调整检测到的关键点的采样窗口
    6.25f is default and fits for KAZE, SURF detected keypoints window ratio
    //6.25f是默认值，适用于KAZE, SURF检测到的关键点窗口比率
    6.75f should be the scale for SIFT detected keypoints window ratio
    //6.75f应为SIFT检测关键点窗口比的尺度
    5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
    //5.00f应为AKAZE、MSD、AGAST、FAST、BRISK关键点窗口比的刻度
    0.75f should be the scale for ORB keypoints ratio
    //0.75f应该是ORB关键点比率的刻度
    1.50f was the default in original implementation
    //1.50f是原始实现中的默认值
    */
    BEBLID(int n_wls = 512, float scale_factor = 1);

    /** @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
    (second variant).

    @param image Image.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    keypoints输入关键点集合。描述符不能用于的关键点
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    计算机被移除。有时可以添加新的关键点，例如:SIFT重复关键点有几个主导方向(对于每个方向)。
    @param  descriptors Computed descriptors. 计算描述符。
            In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i].
            在方法的第二个变体中，描述符[i]是为关键点[i]计算的描述符
            Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
            第j行是关键点(或keypoints[i])是关键点第j个关键点的描述符。
     */
    CV_WRAP void compute(cv::InputArray image, CV_OUT CV_IN_OUT std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

    CV_WRAP int descriptorSize() const override;
    CV_WRAP int descriptorType() const override;
    CV_WRAP int defaultNorm() const override;

    //! Return true if detector object is empty
    CV_WRAP bool empty() const override;
    CV_WRAP cv::String getDefaultName() const override;

    /** @brief Creates the BEBLID descriptor.

    @param n_wls The number of final weak-learners in the descriptor. It must be a multiple of 8 such as 256 or 512.
    @param 描述符中最终弱学习器的数量。必须是8的整数倍，例如256或512。
    @param scale_factor Adjusts the sampling window of detected keypoints
    @param scale_factor调整检测到的关键点的采样窗口
    6.25f is default and fits for KAZE, SURF detected keypoints window ratio
    6.75f should be the scale for SIFT detected keypoints window ratio
    5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
    0.75f should be the scale for ORB keypoints ratio
    1.50f was the default in original implementation
    */
    static cv::Ptr<BEBLID> create(int n_wls = 512, float scale_factor = 1);

    // Struct containing the 6 parameters that define an Average Box weak-learner
    // 结构体包含定义平均盒弱学习器的6个参数
    struct ABWLParams {
        int x1, y1, x2, y2, boxRadius, th;
    };

private:
    void computeBEBLID(const cv::Mat& integralImg, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    std::vector<ABWLParams> wl_params_;
    float scale_factor_ = 1;
    cv::Size patch_size_;
};

