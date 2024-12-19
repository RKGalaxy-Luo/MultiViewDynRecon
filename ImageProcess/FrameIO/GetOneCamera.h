/*****************************************************************//**
 * \file   GetOneCamera.h
 * \brief  ��ò�����ÿһ������ͷ
 * 
 * \author LUO
 * \date   January 19th 2024
 *********************************************************************/
#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Frame.hpp>
#include <libobsensor/hpp/Device.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/StreamProfile.hpp>
#include <base/ThreadPool.h>
#include <base/GlobalConfigs.h>

class GetOneCamera {

public:

	/**
	 * \brief ����һ������ͷ��pipe�������Ӧ����ͷ����.
	 * 
	 * \param AStreamPipe ���뵱ǰ����ͷ��pipe
	 * \param PipeID ���뵱ǰ����ͷ��ID
	 * \param CameraThread �̳߳�
	 */
    GetOneCamera(std::shared_ptr<ob::Pipeline> AStreamPipe, int PipeID , std::shared_ptr<ThreadPool> CameraThread);

    /**
     * \brief �������ݼ���·������Ӧ����ı��.
     * 
     * \param DataPath ���ݼ�·��
     * \param cameraID ������
     * \param CameraThread �߳�
     */
    GetOneCamera(std::string DataPath, int cameraID, std::shared_ptr<ThreadPool> CameraThread);
    ~GetOneCamera() {

    }


    /**
     * \brief ��ȡ��ǰRGBͼ��.
     *
     * \return ��ǰ��RGBͼ��
     */
    cv::Mat GetCurrentColorImage();

    /**
     * \brief ��ȡ��ǰ��Depthͼ��.
     *
     * \return ��ǰDepthͼ��
     */
    cv::Mat GetCurrentDepthImage();

    /**
     * \brief ��ȡ��ǰ��IRͼ��.
     *
     * \return ��ǰIRͼ��
     */
    cv::Mat GetCurrentIRImage();

    /**
     * \brief ��ò�ͼ���ڵ�����.
     * 
     * \return RGB���ڵ�����.
     */
    cv::String GetColorWindowName();

    /**
     * \brief ������ͼ���ڵ�����.
     * 
     * \return RGB���ͼ������
     */
    cv::String GetDepthWindowName();

    /**
     * \brief �������ͼ���Ѿ�׼������.
     * 
     * \return ���ͼ���Ƿ�׼������
     */
    bool imagesReady() {
        if (colorPrepared == true && depthPrepared == true) return true;
        else return false;
    }

    /**
     * \brief ֹͣCamera.
     *
     */
    void CameraStop();

private:
    std::shared_ptr<ThreadPool> pool;
	std::shared_ptr<ob::Pipeline> pipe;                         // �������ͷ��pipe
	int CameraID;                                               // �������ͷ��ID
    std::mutex frameMutex;                                      // ��ס�������ͷ֡����
    std::mutex imageMutex;                                      // ��סͼ���ȡ����
    std::mutex showMutex;                                       // ��ʾͼ�����
    std::condition_variable activateShowThread;                 // ������ʾ�̵߳�����
    cv::Mat ColorImage;                                         // ���cv::Mat���͵�RGBͼ��
    cv::Mat DepthImage;                                         // ���cv::Mat���͵�Depthͼ��
    std::vector<std::shared_ptr<ob::Frame>> Frames;             // ͼ��Ļ�����
    std::vector<std::shared_ptr<ob::Frame>> srcFrames;          // ͼ��Ļ�����
    std::shared_ptr<ob::Frame> colorFrames;                     // RGBͼ��
    std::shared_ptr<ob::Frame> depthFrames;                     // ���ͼ��

    cv::Mat CurrentColorImage;                                  // ��ǰRGBͼ��
    cv::Mat CurrentDepthImage;                                  // ��ǰDepthͼ��
    cv::Mat CurrentIRImage;                                     // ��ǰIRͼ��

    std::atomic<bool> stop = false;                             // �����ж��Ƿ�ֹͣ
    std::atomic<bool> colorPrepared = false;                    // �ж�RGBͼ���Ƿ�׼������
    std::atomic<bool> depthPrepared = false;                    // �ж�Depthͼ���Ƿ�׼������

    cv::String ColorImageName;                                  // ��ʾRGB���ڵ�����
    cv::String DepthImageName;                                  // ��ʾDepth���ڵ�����



    /**
     * \brief ����֡���̣߳���vector�д�����֡���ϲ���Frames���У�������convertToImage()��Frames��ת����cv::Mat.
     * 
     */
    void processFrames();

    /**
     * \brief ��֡ת����cv::Mat.
     * 
     */
    void convertToImage();

    /**
     * \brief �����ݴ�֡��ʽת����cv::Mat��ʽ�������浽��Ӧ��CurrentColorImage��CurrentDepthImage��CurrentIRImage .
     * 
     */
    void FrameToMat(std::shared_ptr<ob::Frame> frame);


};
