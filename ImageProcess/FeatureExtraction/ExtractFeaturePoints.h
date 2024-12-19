#pragma once
#include "BEBLID.h"
#include <base/Logging.h>
#include <base/GlobalConfigs.h>
class ExtractFeaturePoints
{
public:
    /**
     * \brief ORB��ȡƥ���
     * \param ����ͼ��1
     * \param ����ͼ��2
     * \param ƥ���1
     * \param ƥ���2
     */
    void ExtractFeaturePointsORB(cv::Mat Image1, cv::Mat Image2, std::vector<cv::Point2f>& matchpoints1, std::vector<cv::Point2f>& matchpoints2);

    /**
     * \brief BEBLID��ȡƥ���
     * \param ����ͼ��1
     * \param ����ͼ��2
     * \param ƥ���1
     * \param ƥ���2
     */
    void ExtractFeaturePointsBEBLID(cv::Mat Image1, cv::Mat Image2, std::vector<cv::Point2f>& matchpoints1, std::vector<cv::Point2f>& matchpoints2);
};


