#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>												//��׼C++���е�������������ͷ�ļ���
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include <utility>
#include <chrono>
class BEBLID : public cv::Feature2D
{
public:
    /** @brief Creates the BEBLID descriptor.
    //����BEBLID��������
    @param n_wls The number of final weak-learners in the descriptor. It must be a multiple of 8 such as 256 or 512.
    //��������������ѧϰ����������������8��������������256��512��
    @param scale_factor Adjusts the sampling window of detected keypoints
    //scale_factor������⵽�Ĺؼ���Ĳ�������
    6.25f is default and fits for KAZE, SURF detected keypoints window ratio
    //6.25f��Ĭ��ֵ��������KAZE, SURF��⵽�Ĺؼ��㴰�ڱ���
    6.75f should be the scale for SIFT detected keypoints window ratio
    //6.75fӦΪSIFT���ؼ��㴰�ڱȵĳ߶�
    5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
    //5.00fӦΪAKAZE��MSD��AGAST��FAST��BRISK�ؼ��㴰�ڱȵĿ̶�
    0.75f should be the scale for ORB keypoints ratio
    //0.75fӦ����ORB�ؼ�����ʵĿ̶�
    1.50f was the default in original implementation
    //1.50f��ԭʼʵ���е�Ĭ��ֵ
    */
    BEBLID(int n_wls = 512, float scale_factor = 1);

    /** @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
    (second variant).

    @param image Image.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    keypoints����ؼ��㼯�ϡ��������������ڵĹؼ���
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    ��������Ƴ�����ʱ��������µĹؼ��㣬����:SIFT�ظ��ؼ����м�����������(����ÿ������)��
    @param  descriptors Computed descriptors. ������������
            In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i].
            �ڷ����ĵڶ��������У�������[i]��Ϊ�ؼ���[i]�����������
            Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
            ��j���ǹؼ���(��keypoints[i])�ǹؼ����j���ؼ������������
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
    @param ��������������ѧϰ����������������8��������������256��512��
    @param scale_factor Adjusts the sampling window of detected keypoints
    @param scale_factor������⵽�Ĺؼ���Ĳ�������
    6.25f is default and fits for KAZE, SURF detected keypoints window ratio
    6.75f should be the scale for SIFT detected keypoints window ratio
    5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
    0.75f should be the scale for ORB keypoints ratio
    1.50f was the default in original implementation
    */
    static cv::Ptr<BEBLID> create(int n_wls = 512, float scale_factor = 1);

    // Struct containing the 6 parameters that define an Average Box weak-learner
    // �ṹ���������ƽ������ѧϰ����6������
    struct ABWLParams {
        int x1, y1, x2, y2, boxRadius, th;
    };

private:
    void computeBEBLID(const cv::Mat& integralImg, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    std::vector<ABWLParams> wl_params_;
    float scale_factor_ = 1;
    cv::Size patch_size_;
};

