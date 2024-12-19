/*****************************************************************//**
 * \file   GetOneCamera.h
 * \brief  获得并配置每一个摄像头
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
	 * \brief 传入一个摄像头的pipe，构造对应摄像头对象.
	 * 
	 * \param AStreamPipe 传入当前摄像头的pipe
	 * \param PipeID 传入当前摄像头的ID
	 * \param CameraThread 线程池
	 */
    GetOneCamera(std::shared_ptr<ob::Pipeline> AStreamPipe, int PipeID , std::shared_ptr<ThreadPool> CameraThread);

    /**
     * \brief 传入数据集的路径，对应相机的编号.
     * 
     * \param DataPath 数据集路径
     * \param cameraID 相机编号
     * \param CameraThread 线程
     */
    GetOneCamera(std::string DataPath, int cameraID, std::shared_ptr<ThreadPool> CameraThread);
    ~GetOneCamera() {

    }


    /**
     * \brief 获取当前RGB图像.
     *
     * \return 当前的RGB图像
     */
    cv::Mat GetCurrentColorImage();

    /**
     * \brief 获取当前的Depth图像.
     *
     * \return 当前Depth图像
     */
    cv::Mat GetCurrentDepthImage();

    /**
     * \brief 获取当前的IR图像.
     *
     * \return 当前IR图像
     */
    cv::Mat GetCurrentIRImage();

    /**
     * \brief 获得彩图窗口的名字.
     * 
     * \return RGB窗口的名字.
     */
    cv::String GetColorWindowName();

    /**
     * \brief 获得深度图窗口的名字.
     * 
     * \return RGB深度图的名字
     */
    cv::String GetDepthWindowName();

    /**
     * \brief 返回相机图像已经准备就绪.
     * 
     * \return 相机图像是否都准备就绪
     */
    bool imagesReady() {
        if (colorPrepared == true && depthPrepared == true) return true;
        else return false;
    }

    /**
     * \brief 停止Camera.
     *
     */
    void CameraStop();

private:
    std::shared_ptr<ThreadPool> pool;
	std::shared_ptr<ob::Pipeline> pipe;                         // 这个摄像头的pipe
	int CameraID;                                               // 这个摄像头的ID
    std::mutex frameMutex;                                      // 锁住这个摄像头帧的锁
    std::mutex imageMutex;                                      // 锁住图像读取的锁
    std::mutex showMutex;                                       // 显示图像的锁
    std::condition_variable activateShowThread;                 // 控制显示线程的阻塞
    cv::Mat ColorImage;                                         // 获得cv::Mat类型的RGB图像
    cv::Mat DepthImage;                                         // 获得cv::Mat类型的Depth图像
    std::vector<std::shared_ptr<ob::Frame>> Frames;             // 图像的缓冲区
    std::vector<std::shared_ptr<ob::Frame>> srcFrames;          // 图像的缓冲区
    std::shared_ptr<ob::Frame> colorFrames;                     // RGB图像
    std::shared_ptr<ob::Frame> depthFrames;                     // 深度图像

    cv::Mat CurrentColorImage;                                  // 当前RGB图像
    cv::Mat CurrentDepthImage;                                  // 当前Depth图像
    cv::Mat CurrentIRImage;                                     // 当前IR图像

    std::atomic<bool> stop = false;                             // 用来判断是否停止
    std::atomic<bool> colorPrepared = false;                    // 判断RGB图像是否准备好了
    std::atomic<bool> depthPrepared = false;                    // 判断Depth图像是否准备好了

    cv::String ColorImageName;                                  // 显示RGB窗口的名字
    cv::String DepthImageName;                                  // 显示Depth窗口的名字



    /**
     * \brief 处理帧的线程，将vector中传来的帧，合并入Frames流中，并唤醒convertToImage()将Frames流转换成cv::Mat.
     * 
     */
    void processFrames();

    /**
     * \brief 将帧转换成cv::Mat.
     * 
     */
    void convertToImage();

    /**
     * \brief 将数据从帧格式转换成cv::Mat格式，并储存到对应的CurrentColorImage、CurrentDepthImage、CurrentIRImage .
     * 
     */
    void FrameToMat(std::shared_ptr<ob::Frame> frame);


};
