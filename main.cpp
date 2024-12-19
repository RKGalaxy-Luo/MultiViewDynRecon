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
	if (res == "640_400" && FRAME_WIDTH != 640) LOGGING(FATAL) << "�㷨�ֱ��������ݼ��ֱ��ʲ�����";
	else if(res == "1280_720" && FRAME_WIDTH != 1280) LOGGING(FATAL) << "�㷨�ֱ��������ݼ��ֱ��ʲ�����";

	size_t totalTime = 0;
	bool intoBuffer = false;

	unsigned int maxFramesNum = 0;
	if (speed == "slow") maxFramesNum = 1000;
	else if (speed == "fast") maxFramesNum = 500;
	else LOGGING(FATAL) << "MaxFramesNum ���ó���";

	AlgorithmSerial Fusion(pool, intoBuffer);		// �㷨����
	const unsigned int beginIndex = 0;					// �㷨�ӵڼ�֡��ʼ
	if (beginIndex >= maxFramesNum) LOGGING(FATAL) << "beginIndex ���ô���";
	size_t frameIndex = beginIndex;						// ��¼֡��

	Fusion.LoadImages(intoBuffer);
	Fusion.setFrameIndex(frameIndex, beginIndex);

	for (int frameIndex = beginIndex; frameIndex < maxFramesNum; frameIndex++) {
		CHECKCUDA(cudaDeviceSynchronize());		// ��ʼǰȷ��ͬ��
		auto start = std::chrono::high_resolution_clock::now();		// ��ȡ����ʼʱ���
		Fusion.setFrameIndex(frameIndex);							// �㷨����֮ǰ�ȱ�ǵ�ǰ�ǵڼ�֡
		if (frameIndex == beginIndex) {
			//std::cout << "�������������������������������� " << frameIndex << " ֡������������������������������" << std::endl;
			Fusion.ProcessFirstFrame();
		}
		else if (frameIndex < maxFramesNum) {
			//std::cout << "�������������������������������� " << frameIndex << " ֡������������������������������" << std::endl;
			Fusion.ProcessFrameStream(false, true);
		}
		CHECKCUDA(cudaDeviceSynchronize());		// ����ʱ��ǰ����ͬ��
		auto end = std::chrono::high_resolution_clock::now();// ��ȡ�������ʱ���
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// �����������ʱ�䣨���룩
		//std::cout << "ͼ�����崦��" << duration << " ����" << std::endl;
		totalTime += duration;
		cv::waitKey(3);
	}
	float averageTime = totalTime * 1.0f / maxFramesNum;
	printf("\n******************************   ÿ֡ƽ��ʱ�� = %.3f ms   ******************************\n", averageTime);
	exit(-1);
	return 0;
}