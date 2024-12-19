/*****************************************************************//**
 * \file   GetImageFromLocal.h
 * \brief  �ӱ������ݼ��л��ͼ��������Ϣ
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
	 * \brief �������Ҫ�Ǵӱ������ݼ�����ȡ����.
	 */
	class GetImageFromLocal {
	private:
		const int deviceCount = MAX_CAMERA_COUNT;
		std::shared_ptr<ThreadPool> pool;

		std::vector<cv::String> depthsPath; // �洢�������ͼ��vector (ǰ����һ���Զ�ͼ�������ĳɶ��̣߳��㷨һ���̣߳���ͼһ���߳�) 
		std::vector<cv::String> imagesPath; // �洢����RGBͼ��vector  (ǰ����һ���Զ�ͼ�������ĳɶ��̣߳��㷨һ���̣߳���ͼһ���߳�)
		std::size_t depthImagesCount = 0;// ��¼һ���ж��������ͼ
		std::size_t colorImagesCount = 0;// ��¼һ���ж�����RGBͼ
		std::vector<cv::Mat> depths;// �洢���ͼ���vector
		std::vector<cv::Mat> colors;// �洢RGBͼ��vector
	public: 

		using Ptr = std::shared_ptr<GetImageFromLocal>;

		/**
		 * \brief �չ��캯��.
		 */
		GetImageFromLocal() {};
		/**
		 * \brief Ĭ����������.
		 * 
		 */
		~GetImageFromLocal() = default;

		/**
		 * \brief ��ȡ���ݼ�������������ĸ���.
		 * 
		 * \return ���ݼ�������������ĸ���
		 */
		int getDeviceCount() {
			return deviceCount;
		}

		/**
		 * \brief ��ȡ��frameNum֡��RGBͼƬ.
		 * 
		 * \param frameNum ֡����
		 * \return ��frameNum֡��RGBͼƬ.
		 */
		cv::Mat GetColorImage(unsigned int frameNum) {
			return colors[frameNum];
		}

		/**
		 * \brief ��ȡ��frameNum֡��DepthͼƬ.
		 * 
		 * \param frameNum ֡����
		 * \return ��frameNum֡��DepthͼƬ.
		 */
		cv::Mat GetDepthImage(unsigned int frameNum) {
			return depths[frameNum];
		}
	};
}
