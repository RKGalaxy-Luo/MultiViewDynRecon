#pragma once
#include "BEBLID.h"
#include <base/Logging.h>
#include <base/GlobalConfigs.h>
class ExtractFeaturePoints
{
public:
    /**
     * \brief ORB提取匹配点
     * \param 输入图像1
     * \param 输入图像2
     * \param 匹配点1
     * \param 匹配点2
     */
    void ExtractFeaturePointsORB(cv::Mat Image1, cv::Mat Image2, std::vector<cv::Point2f>& matchpoints1, std::vector<cv::Point2f>& matchpoints2);

    /**
     * \brief BEBLID提取匹配点
     * \param 输入图像1
     * \param 输入图像2
     * \param 匹配点1
     * \param 匹配点2
     */
    void ExtractFeaturePointsBEBLID(cv::Mat Image1, cv::Mat Image2, std::vector<cv::Point2f>& matchpoints1, std::vector<cv::Point2f>& matchpoints2);
};


