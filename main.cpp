#include <iostream>
#include <conio.h>
#include <glad/glad.h>
#include <core/AlgorithmSerial.h>
#include <opencv2/opencv.hpp>
#include <base/ThreadPool.h>
#include <base/ConfigParser.h>
#include <ImageProcess/FrameIO/GetFrameFromCamera.h>
#include <Windows.h>
using namespace SparseSurfelFusion;

int main(int argc, char** argv) {
	std::shared_ptr<ThreadPool> pool = std::make_shared<ThreadPool>(MAX_THREADS);
	std::string dataPrefixPath = DATA_PATH_PREFIX;
	std::string dis, res, speed, action;
	Constants::ReadDatasetParameter(dataPrefixPath, res, dis, speed, action);
	if (res == "640_400" && FRAME_WIDTH != 640) LOGGING(FATAL) << "算法分辨率与数据集分辨率不符合";
	else if(res == "1280_720" && FRAME_WIDTH != 1280) LOGGING(FATAL) << "算法分辨率与数据集分辨率不符合";

	size_t totalTime = 0;
	bool intoBuffer = false;

	unsigned int maxFramesNum = 0;
	if (speed == "slow") maxFramesNum = 1000;
	else if (speed == "fast") maxFramesNum = 500;
	else LOGGING(FATAL) << "MaxFramesNum 设置出错";

	AlgorithmSerial Fusion(pool, intoBuffer);		// 算法处理
	const unsigned int beginIndex = 0;					// 算法从第几帧开始
	if (beginIndex >= maxFramesNum) LOGGING(FATAL) << "beginIndex 设置错误";
	size_t frameIndex = beginIndex;						// 记录帧数

	Fusion.LoadImages(intoBuffer);
	Fusion.setFrameIndex(frameIndex, beginIndex);

	for (int frameIndex = beginIndex; frameIndex < maxFramesNum; frameIndex++) {
		CHECKCUDA(cudaDeviceSynchronize());		// 开始前确保同步
		auto start = std::chrono::high_resolution_clock::now();		// 获取程序开始时间点
		Fusion.setFrameIndex(frameIndex);							// 算法处理之前先标记当前是第几帧
		if (frameIndex == beginIndex) {
			//std::cout << "···············第 " << frameIndex << " 帧···············" << std::endl;
			Fusion.ProcessFirstFrame();
		}
		else if (frameIndex < maxFramesNum) {
			//std::cout << "···············第 " << frameIndex << " 帧···············" << std::endl;
			Fusion.ProcessFrameStream(false, true);
		}
		CHECKCUDA(cudaDeviceSynchronize());		// 计算时间前计算同步
		auto end = std::chrono::high_resolution_clock::now();// 获取程序结束时间点
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// 计算程序运行时间（毫秒）
		//std::cout << "图像整体处理：" << duration << " 毫秒" << std::endl;
		totalTime += duration;
		cv::waitKey(3);
	}
	float averageTime = totalTime * 1.0f / maxFramesNum;
	printf("\n******************************   每帧平均时间 = %.3f ms   ******************************\n", averageTime);
	exit(-1);
	return 0;
}