/*****************************************************************//**
 * \file   Constants.h
 * \brief  维护主机存取常量，算法中一些常量的结构体，基本常数(算法的超参数).
 * 
 * \author LUO
 * \date  January 29th 2024
 *********************************************************************/
#pragma once
#include <iostream>
#include <base/ConfigParser.h>
#include <base/CommonTypes.h>
#include <math/VectorUtils.h>
#include <math/MatUtils.h>
#include <mesh/MeshConfigs.h>
namespace SparseSurfelFusion {
	
	struct Constants {

		const static float kFilterSigmaS;						// 双边滤波：空间距离权重
		const static float kFilterSigmaR;						// 双边滤波：像素值权重
		
		const static float kForegroundSigma;					// 前景滤波的sigma值

		const static int superSampleScale;						// 渲染图放大倍数(因为相邻两个像素可能投影到渲染图中同一个像素点，因此需要超采样)

		const static unsigned int maxSurfelsNum;				// 最大面元数量

		const static unsigned int maxMeshTrianglesNum;			// 最大网格三角形数量

		const static unsigned int maxNodesNum;					// 最大的节点数

		const static unsigned nodesGraphNeigboursNum;			// 节点图邻居的数量

		const static unsigned int maxSubsampledSparsePointsNum;	// 每个相机下采样最大的下采样点的数量

		const static unsigned int maxSubsampleFrom;				// 从kMaxSubsampleFrom候选节点中最多选择1个节点

		const static float maxCompactedInputPointsScale;		// 对原始稠密点压缩尺度

		const static float NodeRadius;							// 节点平均距离(单位：m)
		const static float VoxelSize;							// 体素大小(单位：m)
		const static float NodeSamplingRadius;					// 节点采样距离(单位：m)

		const static float TSDFThreshold;						// 融合不同相机曲面的TSDF截断阈值(单位：m)
		const static unsigned int MaxOverlappingSurfel;			// 最大融合面元的数量
		const static float MaxTruncatedSquaredDistance;			// 最大截断距离平方，如果融合两点超过这个距离，则不对这两个点融合

		const static int kNumGlobalSolverItarations;
		const static bool kUseElasticPenalty;
		const static int kNumGaussNewtonIterations;

		//稳定面元的置信阈值
		const static int kStableSurfelConfidenceThreshold;

		const static int kRenderingRecentTimeThreshold;

		const static unsigned kMaxMatchedSparseFeature;

		const static unsigned kMaxNumNodePairs;
		const static unsigned kMaxNumSurfelCandidates;

		const static unsigned MaxCrossViewMatchedPairs;
		const static float CrossViewPairsSquaredDisThreshold;					// 跨镜匹配的两个点的距离(平方)阈值

		const static float3 InterpolatedCameraeRotation[MAX_CAMERA_COUNT];		// 相机之间旋转角度的插值(顺序为01,12,20)
		const static float3 InterpolatedCameraeTranslation[MAX_CAMERA_COUNT];	// 相机之间平移的插值(顺序为01,12,20)
		const static mat34 InterpolatedCameraSE3[MAX_CAMERA_COUNT];				// 插值相机的相对于0号相机的位置(顺序为01,12,20)


		/**
		 * \brief 函数是直接从文件里获取，该函数尽可能在构造函数中调用.
		 * 
		 * \param cameraID 相机ID
		 * \return 相机位姿
		 */
		static mat34 GetInitialCameraSE3(const unsigned int cameraID) {
			if (cameraID >= MAX_CAMERA_COUNT) LOGGING(FATAL) << "位姿获取大于相机数量";
			std::string InitialCameraPosePath = DATA_PATH_PREFIX;
			std::string dis, res, speed, action;
			ReadDatasetParameter(InitialCameraPosePath, res, dis, speed, action);
			InitialCameraPosePath = InitialCameraPosePath + "/" + dis + "/" + "CameraPose_" + dis + ".txt";
			std::ifstream infile(InitialCameraPosePath); // 打开名为input.txt的文件
			std::vector<float> floatVector; // 创建一个vector，并将浮点数存储在其中
			float value;
			int lineIdx = 1;
			char buffer[1024];
			while (infile.getline(buffer, sizeof(buffer))) {// 循环读取文件中的浮点数
				if (lineIdx == cameraID * 5 + 1 || lineIdx == cameraID * 5 + 2 || lineIdx == cameraID * 5 + 3) {
					while (infile >> value) {
						floatVector.push_back(value);
					}
					lineIdx++;
					continue;
				}
				lineIdx++;
			}
			infile.close(); // 关闭文件
			mat34 InitialCameraSE3;
			InitialCameraSE3.rot.m00() = floatVector[0];
			InitialCameraSE3.rot.m01() = floatVector[1];
			InitialCameraSE3.rot.m02() = floatVector[2];
			InitialCameraSE3.trans.x   = floatVector[3];
			InitialCameraSE3.rot.m10() = floatVector[4];
			InitialCameraSE3.rot.m11() = floatVector[5];
			InitialCameraSE3.rot.m12() = floatVector[6];
			InitialCameraSE3.trans.y   = floatVector[7];
			InitialCameraSE3.rot.m20() = floatVector[8];
			InitialCameraSE3.rot.m21() = floatVector[9];
			InitialCameraSE3.rot.m22() = floatVector[10];
			InitialCameraSE3.trans.z   = floatVector[11];
			return InitialCameraSE3;
		}

		static mat34 GetInterpolatedCameraSE3(const unsigned int index) {
			if (index >= MAX_CAMERA_COUNT) LOGGING(FATAL) << "位姿获取大于相机数量";
			std::string InitialCameraPosePath = DATA_PATH_PREFIX;
			std::string dis, res, speed, action;
			ReadDatasetParameter(InitialCameraPosePath, res, dis, speed, action);
			InitialCameraPosePath = InitialCameraPosePath + "/" + dis + "/" + "CameraPose_" + dis + ".txt";
			std::ifstream infile(InitialCameraPosePath); // 打开名为input.txt的文件
			std::vector<float> floatVector; // 创建一个vector，并将浮点数存储在其中
			float value;
			int lineIdx = 1;
			char buffer[1024];
			while (infile.getline(buffer, sizeof(buffer))) {// 循环读取文件中的浮点数
				if (lineIdx == index * 5 + 16 || lineIdx == index * 5 + 17 || lineIdx == index * 5 + 18) {
					while (infile >> value) {
						floatVector.push_back(value);
					}
					lineIdx++;
					continue;
				}
				lineIdx++;
			}
			infile.close(); // 关闭文件
			mat34 InterpolatedCameraSE3;
			InterpolatedCameraSE3.rot.m00() = floatVector[0];
			InterpolatedCameraSE3.rot.m01() = floatVector[1];
			InterpolatedCameraSE3.rot.m02() = floatVector[2];
			InterpolatedCameraSE3.trans.x   = floatVector[3];
			InterpolatedCameraSE3.rot.m10() = floatVector[4];
			InterpolatedCameraSE3.rot.m11() = floatVector[5];
			InterpolatedCameraSE3.rot.m12() = floatVector[6];
			InterpolatedCameraSE3.trans.y   = floatVector[7];
			InterpolatedCameraSE3.rot.m20() = floatVector[8];
			InterpolatedCameraSE3.rot.m21() = floatVector[9];
			InterpolatedCameraSE3.rot.m22() = floatVector[10];
			InterpolatedCameraSE3.trans.z   = floatVector[11];
			return InterpolatedCameraSE3;
		}

		static bool GetOnlineForegroundSegmenter() {
			std::string prefixPath = DATA_PATH_PREFIX;
			std::ifstream file(prefixPath + "/" + "数据集参数配置.txt");
			std::string line;
			std::string Segmenter = "";
			if (file.is_open()) {
				while (std::getline(file, line)) {
					if (line.find("OnlineForegroundSegmenter:") == 0) {
						Segmenter = line.substr(26); // 提取 "Action: " 后面的部分
					}
				}
				file.close();
			}
			else {
				std::cerr << "无法打开文件" << std::endl;
				return false;
			}
			if (Segmenter == "on" || Segmenter == "ON") {
				return true;
			}
			else if (Segmenter == "off" || Segmenter == "OFF") {
				return false;
			}
			else LOGGING(INFO) << "Dataset Parameter Segmenter Error!";
			return false;
		}

		static bool ShowInitialCameraPose(const unsigned int cameraID) {
			if (cameraID >= MAX_CAMERA_COUNT) {
				LOGGING(FATAL) << "位姿获取大于相机数量";
				return false;
			}
			else {
				mat34 InitialCameraSE3 = GetInitialCameraSE3(cameraID);
				printf("Camera_%d 初始化位姿:\n", cameraID);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InitialCameraSE3.rot.m00(), InitialCameraSE3.rot.m01(), InitialCameraSE3.rot.m02(), InitialCameraSE3.trans.x);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InitialCameraSE3.rot.m10(), InitialCameraSE3.rot.m11(), InitialCameraSE3.rot.m12(), InitialCameraSE3.trans.y);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InitialCameraSE3.rot.m20(), InitialCameraSE3.rot.m21(), InitialCameraSE3.rot.m22(), InitialCameraSE3.trans.z);
				return true;
			}
		}

		static bool ShowInterpolatedCameraPose(const unsigned int index) {
			if (index >= MAX_CAMERA_COUNT) {
				LOGGING(FATAL) << "位姿获取大于相机数量";
				return false;
			}
			else {
				mat34 InterpolatedCameraSE3 = GetInterpolatedCameraSE3(index);
				if (index != 2) printf("InterpolatedCamera_%d_%d 位姿变换矩阵:\n", index, index + 1);
				else printf("InterpolatedCamera_%d_%d 位姿变换矩阵:\n", index, 0);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InterpolatedCameraSE3.rot.m00(), InterpolatedCameraSE3.rot.m01(), InterpolatedCameraSE3.rot.m02(), InterpolatedCameraSE3.trans.x);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InterpolatedCameraSE3.rot.m10(), InterpolatedCameraSE3.rot.m11(), InterpolatedCameraSE3.rot.m12(), InterpolatedCameraSE3.trans.y);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InterpolatedCameraSE3.rot.m20(), InterpolatedCameraSE3.rot.m21(), InterpolatedCameraSE3.rot.m22(), InterpolatedCameraSE3.trans.z);
				return true;
			}
		}

		static void ReadDatasetParameter(std::string prefixPath, std::string& res, std::string& dis, std::string& speed, std::string& act) {
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

		//网格化
		const static int LUTparent_Host[8][27];

		const static int LUTchild_Host[8][27];

		static const int markOffset_Host = 31;

		static const int maxDepth_Host = MAX_DEPTH_OCTREE;
	};

}
