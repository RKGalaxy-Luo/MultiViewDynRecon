/*****************************************************************//**
 * \file   GetFrameFromCamera.h
 * \brief  从摄像头实时的视频流中获得图像和深度信息
 * 
 * \author LUO
 * \date   January 15th 2024
 *********************************************************************/
#pragma once
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <queue>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
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
#include <base/GlobalConfigs.h>
#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/ThreadPool.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>
#include <ImageProcess/FrameIO/GetOneCamera.h>
#include <ImageProcess/FeatureExtraction/ExtractFeaturePoints.h>

namespace SparseSurfelFusion {


    /**
     * \brief 从摄像头实时的视频流中获得图像和深度信息.
     */
	class GetFrameFromCamera {
    public:
        /**
         * \brief 提取特征点的方式.
         */
        enum class ExtractFeatureMethod {
            ORB,
            BEBLID
        };

    private:
        std::shared_ptr<ThreadPool> pool;                                               // 线程池对象
        bool CaptureCamera = true;                                                      // 判断是否停止捕获图片
        int devCount = 0;                                                               // 获得一共多少设备
        size_t FrameIndex = 0;                                                          // 记录此时是第几帧
        std::shared_ptr<GetOneCamera>       ACamera[MAX_CAMERA_COUNT];                  // 一个相机对象，处理一个相机所有事务
        std::shared_ptr<ob::DeviceInfo>     ACameraInfo[MAX_CAMERA_COUNT];              // 记录相机的出厂参数
        OBCameraParam                       ACameraParam[MAX_CAMERA_COUNT];             // 相机参数，畸变系数
        Intrinsic                           ColorIntrinsic[MAX_CAMERA_COUNT];           // RGB图像的内参
        Intrinsic                           DepthIntrinsic[MAX_CAMERA_COUNT];           // Depth图像的内参
        Intrinsic                           ClipColorIntrinsic[MAX_CAMERA_COUNT];       // 剪裁图像的内参

    private:
        /**
         * \brief 剪裁图像，为特征匹配.
         * 
         * \param image 输入原图
         * \return 剪裁后图片
         */
        cv::Mat clipImageBoundary(cv::Mat image) {
            int left = CLIP_BOUNDARY;
            int top = CLIP_BOUNDARY;
            int right = FRAME_WIDTH - CLIP_BOUNDARY;
            int bottom = FRAME_HEIGHT - CLIP_BOUNDARY;

            cv::Rect roi(left, top, right - left, bottom - top);
            cv::Mat clipedImage = image(roi);
            return clipedImage;
        }

        void ReadDatasetParameter(std::string prefixPath, std::string& res, std::string& dis, std::string& speed, std::string& act) {
            std::ifstream file(prefixPath + "/" + "数据集参数配置.txt");
            std::string line;

            if (file.is_open()) {
                while (std::getline(file, line)) {
                    if (line.find("Action:") == 0) {
                        act = line.substr(7); // 提取 "Action: " 后面的部分
                    }
                    else if (line.find("Distance:") == 0) {
                        dis = line.substr(9); // 提取 "Distance: " 后面的部分
                    }
                    else if (line.find("Resolution:") == 0) {
                        res = line.substr(11); // 提取 "Resolution: " 后面的部分
                    }
                    else if (line.find("Speed:") == 0) {
                        speed = line.substr(6); // 提取 "Speed: " 后面的部分
                    }
                }
                file.close();
            }
            else {
                std::cerr << "无法打开文件" << std::endl;
            }
        }
    public:
        using Ptr = std::shared_ptr<GetFrameFromCamera>;

        /**
         * \brief 构造函数，多线程读取多个相机图像.
         * 
         */
        GetFrameFromCamera(std::shared_ptr<ThreadPool> threadPool) : pool(threadPool) {
            CaptureCamera = true;                               // 开始获取图像
            std::string prefixPath = DATA_PATH_PREFIX;
            devCount = MAX_CAMERA_COUNT;
            std::string dis, res, speed, action;
            ReadDatasetParameter(prefixPath, res, dis, speed, action);
            for (int i = 0; i < devCount; i++) {
                std::string path;
                if (action == "calibration") {
                    LOGGING(FATAL) << "不应该使用标定数据集重建";
                }
                else {
                    path = prefixPath + dis + "/" + action + "/" + res + "/" + speed + "/Camera_" + std::to_string(i);
                }
                // 将相机参数赋值给Intrinsic
                readIntrinsicFiles(i, path + "/Intrinsic Parameters.txt");
            }
        }
        /**
         * \brief 析构函数.
         */
        ~GetFrameFromCamera() {

        }

        /**
         * \brief 获得RGB相机的内参.
         * 
         * \param 相机顺序ID
         * \return RGB相机内参数组的指针
         */
        Intrinsic getColorIntrinsic(unsigned int CameraID) {
            return ColorIntrinsic[CameraID];
        }

        /**
         * \brief 获得深度相机的内参.
         * 
         * \param 相机顺序ID
         * \return Depth相机内参数组的指针
         */
        Intrinsic getDepthIntrinsic(unsigned int CameraID) {
            return DepthIntrinsic[CameraID];
        }

        /**
         * \brief 获得剪裁后RGB相机内参.
         * 
         * \param 相机顺序ID
         * \return 剪裁后RGB相机数组的指针
         */
        Intrinsic getClipColorIntrinsic(unsigned int CameraID) {
            return ClipColorIntrinsic[CameraID];
        }

        /**
         * \brief 设置当前的帧ID.
         * 
         * \param index 帧的索引
         */
        void setFrameIndex(size_t index) {
            FrameIndex = index;
        }

        /**
         * \brief 获得当前帧ID.
         */
        size_t getCurrentFrameIndex() {
            return FrameIndex;
        }

        /**
         * \brief 获得相机数量.
         * 
         * \return 相机数量
         */
        int getCameraCount() {
            return devCount;
        }

        /**
         * \brief 获得是否继续从摄像头读取图像.
         * 
         * \param 获得相机是否停止
         * \return 相机停止的标志，false是停止，true是未停止
         */
        bool ContinueCapturingCamera() {
            return CaptureCamera;
        }

        /**
         * \brief 告诉相机停止获取图像.
         *
         */
        void StopIt() {
            CaptureCamera = false;
            for (int i = 0; i < devCount; i++) {
                ACamera[i]->CameraStop();
                std::cout << "第" << i << "号相机结束进程" << std::endl;
            }

        }

        /**
         * \brief 获取当前相机ID的RGB图像.
         * 
         * \param 相机的ID
         * \return 当前ID相机的RGB图像
         */
        cv::Mat GetColorImage(unsigned int CameraID) {
            if (!ACamera[CameraID]->GetCurrentColorImage().empty()) {
                cv::Mat rst = ACamera[CameraID]->GetCurrentColorImage(); 
                return rst;
            }
            else LOGGING(INFO) << "RGB图像为空！";
        }


        /**
         * \brief 获取当前相机ID的Depth图像.
         * 
         * \param 相机的ID
         * \return 当前ID相机的Depth图像
         */
        cv::Mat GetDepthImage(unsigned int CameraID) {
            if (!ACamera[CameraID]->GetCurrentDepthImage().empty()) {
                cv::Mat rst = ACamera[CameraID]->GetCurrentDepthImage();
                return rst;
            }
            else LOGGING(INFO) << "Depth图像为空！";
        }

        /**
         * \brief 获得RGB窗口的名字.
         * 
         * \param CameraID 相机的ID
         * \return RGB窗口的名字.
         */
        cv::String GetColorWindowName(unsigned int CameraID) {
            return ACamera[CameraID]->GetColorWindowName();
        }

        /**
         * \brief 获得Depth窗口的名字.
         * 
         * \param CameraID 相机的ID
         * \return Depth窗口的名字.
         */
        cv::String GetDepthWindowName(unsigned int CameraID) {
            return ACamera[CameraID]->GetDepthWindowName();
        }

        bool CamerasReady() {
            for (int i = 0; i < devCount; i++) {    // 只要有一个相机没有准备好即没准备好
                if (!ACamera[i]->imagesReady()) return false;
            }
            return true;    // 所有相机就绪
        }

        /**
         * \brief 计算剪裁后图像的特征点.
         *
         * \param 上一个相机的图像
         * \param 当前相机的图像
         * \param type 筛选匹配点的方式
         */
        void CalculateFeaturePoints(cv::Mat& previousImage, cv::Mat& currentImage, SynchronizeArray<PixelCoordinate>& matchedPairs, ExtractFeatureMethod type = ExtractFeatureMethod::BEBLID) {
            std::vector<cv::Point2f> previousImageMatchPoints, currentImageMatchPoints;
            cv::Mat clipedPreviousImage = clipImageBoundary(previousImage);
            cv::Mat clipedCurrentImage = clipImageBoundary(currentImage);
            ExtractFeaturePoints extract;
            switch (type)
            {
                case SparseSurfelFusion::GetFrameFromCamera::ExtractFeatureMethod::ORB:
                    extract.ExtractFeaturePointsORB(clipedPreviousImage, clipedCurrentImage, previousImageMatchPoints, currentImageMatchPoints);
                    break;
                case SparseSurfelFusion::GetFrameFromCamera::ExtractFeatureMethod::BEBLID:
                    extract.ExtractFeaturePointsBEBLID(clipedPreviousImage, clipedCurrentImage, previousImageMatchPoints, currentImageMatchPoints);
                    break;
                default:
                    LOGGING(FATAL) << "非法的特征点提取方法";
                    break;
            }
            std::vector<PixelCoordinate>& matchedPoints = matchedPairs.HostArray();     // 暴露指针：previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum)
            for (int i = 0; i < previousImageMatchPoints.size(); i++) {
                unsigned int x = cvRound(previousImageMatchPoints[i].x);
                unsigned int y = cvRound(previousImageMatchPoints[i].y);
                matchedPoints.emplace_back(PixelCoordinate(y, x));
            }
            for (int i = 0; i < currentImageMatchPoints.size(); i++) {
                unsigned int x = cvRound(currentImageMatchPoints[i].x);
                unsigned int y = cvRound(currentImageMatchPoints[i].y);
                matchedPoints.emplace_back(PixelCoordinate(y, x));

            }
            matchedPairs.SynchronizeToDevice(); // 上传到GPU
            CHECKCUDA(cudaDeviceSynchronize()); // 同步
            //std::cout << "matchedPairs Size = " << matchedPairs.DeviceArrayReadOnly().Size() << std::endl;
        }

        void readIntrinsicFiles(const unsigned int CameraID, std::string path) {
            std::ifstream infile(path); // 打开名为input.txt的文件
            std::vector<float> floatVector; // 创建一个vector，并将浮点数存储在其中
            float value;
            int lineIdx = 0;
            char buffer[1024];
            while (infile.getline(buffer, sizeof(buffer))) {// 循环读取文件中的浮点数
                if (lineIdx == 3 || lineIdx == 4 || lineIdx == 5) {
                    while (infile >> value) {
                        floatVector.push_back(value);
                    }
                    lineIdx++;
                    continue;
                }
                lineIdx++;
            }
            infile.close(); // 关闭文件
            Intrinsic intrinsic;
            intrinsic.focal_x = floatVector[0];
            intrinsic.focal_y = floatVector[4];
            intrinsic.principal_x = floatVector[2];
            intrinsic.principal_y = floatVector[5];

            ColorIntrinsic[CameraID] = intrinsic;
            DepthIntrinsic[CameraID] = intrinsic;
            ClipColorIntrinsic[CameraID].focal_x = intrinsic.focal_x;
            ClipColorIntrinsic[CameraID].focal_y = intrinsic.focal_y;
            ClipColorIntrinsic[CameraID].principal_x = intrinsic.principal_x - CLIP_BOUNDARY;
            ClipColorIntrinsic[CameraID].principal_y = intrinsic.principal_y - CLIP_BOUNDARY;
        }

	};
}
