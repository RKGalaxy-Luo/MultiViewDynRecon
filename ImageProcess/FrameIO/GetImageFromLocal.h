/*****************************************************************//**
 * \file   GetImageFromLocal.h
 * \brief  从本地数据集中获得图像和深度信息
 * 
 * \author LUO
 * \date   January 15th 2024
 *********************************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <base/GlobalConfigs.h>
#include <base/Logging.h>
#include <base/ThreadPool.h>
using namespace std;


namespace SparseSurfelFusion {
	/**
	 * \brief 这个类主要是从本地数据集中拿取数据.
	 */
	class GetImageFromLocal {
	private:
		const int deviceCount = MAX_CAMERA_COUNT;
		std::shared_ptr<ThreadPool> pool;

		std::vector<cv::String> depthsPath; // 存储所有深度图的vector (前期先一次性读图，后续改成多线程，算法一个线程，读图一个线程) 
		std::vector<cv::String> imagesPath; // 存储所有RGB图的vector  (前期先一次性读图，后续改成多线程，算法一个线程，读图一个线程)
		std::size_t depthImagesCount = 0;// 记录一共有多少张深度图
		std::size_t colorImagesCount = 0;// 记录一共有多少张RGB图
		std::vector<cv::Mat> depths;// 存储深度图像的vector
		std::vector<cv::Mat> colors;// 存储RGB图的vector
	public: 

		using Ptr = std::shared_ptr<GetImageFromLocal>;

		/**
		 * \brief 空构造函数.
		 */
		GetImageFromLocal() {};
		/**
		 * \brief 默认析构函数.
		 * 
		 */
		~GetImageFromLocal() = default;

		/**
		 * \brief 获取数据集参与拍摄相机的个数.
		 * 
		 * \return 数据集参与拍摄相机的个数
		 */
		int getDeviceCount() {
			return deviceCount;
		}

		/**
		 * \brief 获取第frameNum帧的RGB图片.
		 * 
		 * \param frameNum 帧索引
		 * \return 第frameNum帧的RGB图片.
		 */
		cv::Mat GetColorImage(unsigned int frameNum) {
			return colors[frameNum];
		}

		/**
		 * \brief 获取第frameNum帧的Depth图片.
		 * 
		 * \param frameNum 帧索引
		 * \return 第frameNum帧的Depth图片.
		 */
		cv::Mat GetDepthImage(unsigned int frameNum) {
			return depths[frameNum];
		}
	};
}
