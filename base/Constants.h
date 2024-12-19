/*****************************************************************//**
 * \file   Constants.h
 * \brief  ά��������ȡ�������㷨��һЩ�����Ľṹ�壬��������(�㷨�ĳ�����).
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

		const static float kFilterSigmaS;						// ˫���˲����ռ����Ȩ��
		const static float kFilterSigmaR;						// ˫���˲�������ֵȨ��
		
		const static float kForegroundSigma;					// ǰ���˲���sigmaֵ

		const static int superSampleScale;						// ��Ⱦͼ�Ŵ���(��Ϊ�����������ؿ���ͶӰ����Ⱦͼ��ͬһ�����ص㣬�����Ҫ������)

		const static unsigned int maxSurfelsNum;				// �����Ԫ����

		const static unsigned int maxMeshTrianglesNum;			// �����������������

		const static unsigned int maxNodesNum;					// ���Ľڵ���

		const static unsigned nodesGraphNeigboursNum;			// �ڵ�ͼ�ھӵ�����

		const static unsigned int maxSubsampledSparsePointsNum;	// ÿ������²��������²����������

		const static unsigned int maxSubsampleFrom;				// ��kMaxSubsampleFrom��ѡ�ڵ������ѡ��1���ڵ�

		const static float maxCompactedInputPointsScale;		// ��ԭʼ���ܵ�ѹ���߶�

		const static float NodeRadius;							// �ڵ�ƽ������(��λ��m)
		const static float VoxelSize;							// ���ش�С(��λ��m)
		const static float NodeSamplingRadius;					// �ڵ��������(��λ��m)

		const static float TSDFThreshold;						// �ںϲ�ͬ��������TSDF�ض���ֵ(��λ��m)
		const static unsigned int MaxOverlappingSurfel;			// ����ں���Ԫ������
		const static float MaxTruncatedSquaredDistance;			// ���ضϾ���ƽ��������ں����㳬��������룬�򲻶����������ں�

		const static int kNumGlobalSolverItarations;
		const static bool kUseElasticPenalty;
		const static int kNumGaussNewtonIterations;

		//�ȶ���Ԫ��������ֵ
		const static int kStableSurfelConfidenceThreshold;

		const static int kRenderingRecentTimeThreshold;

		const static unsigned kMaxMatchedSparseFeature;

		const static unsigned kMaxNumNodePairs;
		const static unsigned kMaxNumSurfelCandidates;

		const static unsigned MaxCrossViewMatchedPairs;
		const static float CrossViewPairsSquaredDisThreshold;					// �羵ƥ���������ľ���(ƽ��)��ֵ

		const static float3 InterpolatedCameraeRotation[MAX_CAMERA_COUNT];		// ���֮����ת�ǶȵĲ�ֵ(˳��Ϊ01,12,20)
		const static float3 InterpolatedCameraeTranslation[MAX_CAMERA_COUNT];	// ���֮��ƽ�ƵĲ�ֵ(˳��Ϊ01,12,20)
		const static mat34 InterpolatedCameraSE3[MAX_CAMERA_COUNT];				// ��ֵ����������0�������λ��(˳��Ϊ01,12,20)


		/**
		 * \brief ������ֱ�Ӵ��ļ����ȡ���ú����������ڹ��캯���е���.
		 * 
		 * \param cameraID ���ID
		 * \return ���λ��
		 */
		static mat34 GetInitialCameraSE3(const unsigned int cameraID) {
			if (cameraID >= MAX_CAMERA_COUNT) LOGGING(FATAL) << "λ�˻�ȡ�����������";
			std::string InitialCameraPosePath = DATA_PATH_PREFIX;
			std::string dis, res, speed, action;
			ReadDatasetParameter(InitialCameraPosePath, res, dis, speed, action);
			InitialCameraPosePath = InitialCameraPosePath + "/" + dis + "/" + "CameraPose_" + dis + ".txt";
			std::ifstream infile(InitialCameraPosePath); // ����Ϊinput.txt���ļ�
			std::vector<float> floatVector; // ����һ��vector�������������洢������
			float value;
			int lineIdx = 1;
			char buffer[1024];
			while (infile.getline(buffer, sizeof(buffer))) {// ѭ����ȡ�ļ��еĸ�����
				if (lineIdx == cameraID * 5 + 1 || lineIdx == cameraID * 5 + 2 || lineIdx == cameraID * 5 + 3) {
					while (infile >> value) {
						floatVector.push_back(value);
					}
					lineIdx++;
					continue;
				}
				lineIdx++;
			}
			infile.close(); // �ر��ļ�
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
			if (index >= MAX_CAMERA_COUNT) LOGGING(FATAL) << "λ�˻�ȡ�����������";
			std::string InitialCameraPosePath = DATA_PATH_PREFIX;
			std::string dis, res, speed, action;
			ReadDatasetParameter(InitialCameraPosePath, res, dis, speed, action);
			InitialCameraPosePath = InitialCameraPosePath + "/" + dis + "/" + "CameraPose_" + dis + ".txt";
			std::ifstream infile(InitialCameraPosePath); // ����Ϊinput.txt���ļ�
			std::vector<float> floatVector; // ����һ��vector�������������洢������
			float value;
			int lineIdx = 1;
			char buffer[1024];
			while (infile.getline(buffer, sizeof(buffer))) {// ѭ����ȡ�ļ��еĸ�����
				if (lineIdx == index * 5 + 16 || lineIdx == index * 5 + 17 || lineIdx == index * 5 + 18) {
					while (infile >> value) {
						floatVector.push_back(value);
					}
					lineIdx++;
					continue;
				}
				lineIdx++;
			}
			infile.close(); // �ر��ļ�
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
			std::ifstream file(prefixPath + "/" + "���ݼ���������.txt");
			std::string line;
			std::string Segmenter = "";
			if (file.is_open()) {
				while (std::getline(file, line)) {
					if (line.find("OnlineForegroundSegmenter:") == 0) {
						Segmenter = line.substr(26); // ��ȡ "Action: " ����Ĳ���
					}
				}
				file.close();
			}
			else {
				std::cerr << "�޷����ļ�" << std::endl;
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
				LOGGING(FATAL) << "λ�˻�ȡ�����������";
				return false;
			}
			else {
				mat34 InitialCameraSE3 = GetInitialCameraSE3(cameraID);
				printf("Camera_%d ��ʼ��λ��:\n", cameraID);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InitialCameraSE3.rot.m00(), InitialCameraSE3.rot.m01(), InitialCameraSE3.rot.m02(), InitialCameraSE3.trans.x);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InitialCameraSE3.rot.m10(), InitialCameraSE3.rot.m11(), InitialCameraSE3.rot.m12(), InitialCameraSE3.trans.y);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InitialCameraSE3.rot.m20(), InitialCameraSE3.rot.m21(), InitialCameraSE3.rot.m22(), InitialCameraSE3.trans.z);
				return true;
			}
		}

		static bool ShowInterpolatedCameraPose(const unsigned int index) {
			if (index >= MAX_CAMERA_COUNT) {
				LOGGING(FATAL) << "λ�˻�ȡ�����������";
				return false;
			}
			else {
				mat34 InterpolatedCameraSE3 = GetInterpolatedCameraSE3(index);
				if (index != 2) printf("InterpolatedCamera_%d_%d λ�˱任����:\n", index, index + 1);
				else printf("InterpolatedCamera_%d_%d λ�˱任����:\n", index, 0);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InterpolatedCameraSE3.rot.m00(), InterpolatedCameraSE3.rot.m01(), InterpolatedCameraSE3.rot.m02(), InterpolatedCameraSE3.trans.x);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InterpolatedCameraSE3.rot.m10(), InterpolatedCameraSE3.rot.m11(), InterpolatedCameraSE3.rot.m12(), InterpolatedCameraSE3.trans.y);
				printf("(%.5f, %.5f, %.5f, %.5f)\n", InterpolatedCameraSE3.rot.m20(), InterpolatedCameraSE3.rot.m21(), InterpolatedCameraSE3.rot.m22(), InterpolatedCameraSE3.trans.z);
				return true;
			}
		}

		static void ReadDatasetParameter(std::string prefixPath, std::string& res, std::string& dis, std::string& speed, std::string& act) {
			std::ifstream file(prefixPath + "/" + "���ݼ���������.txt");
			std::string line;

			if (file.is_open()) {
				while (std::getline(file, line)) {
					if (line.find("Action:") == 0) {
						act = line.substr(7); // ��ȡ "Action: " ����Ĳ���
					}
					else if (line.find("Distance:") == 0) {
						dis = line.substr(9); // ��ȡ "Distance: " ����Ĳ���
					}
					else if (line.find("Resolution:") == 0) {
						res = line.substr(11); // ��ȡ "Resolution: " ����Ĳ���
					}
					else if (line.find("Speed:") == 0) {
						speed = line.substr(6); // ��ȡ "Speed: " ����Ĳ���
					}
				}
				file.close();
			}
			else {
				std::cerr << "�޷����ļ�" << std::endl;
			}
		}

		//����
		const static int LUTparent_Host[8][27];

		const static int LUTchild_Host[8][27];

		static const int markOffset_Host = 31;

		static const int maxDepth_Host = MAX_DEPTH_OCTREE;
	};

}
