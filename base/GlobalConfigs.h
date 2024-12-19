/*****************************************************************//**
 * \file   GlobalConfigs.h
 * \brief  ȫ�ֲ���
 * 
 * \author LUO
 * \date   January 6th 2024
 *********************************************************************/
#pragma once


/*	
 * �� CUDA 5.5 �� Eigen 3.3 ��ʼ�������� CUDA �ں���ʹ�� Eigen �ľ���������������ʵ�̶ֹ���С�����ڴ������С����ʱ�ر����á�
 * Ĭ������£��� Eigen �ı�ͷ�������� nvcc ����� .cu �ļ���ʱ������� Eigen �ĺ����ͷ��������� device host �ؼ�����Ϊǰ׺��
 * �Ӷ����Դ��������豸����������ǡ�����ͨ���ڰ����κ� Eigen �ı�ͷ֮ǰ���� EIGEN_NO_CUDA �����ô�֧�֡�
 * �� .cu �ļ�����������ʹ�� Eigen ʱ������������ڽ���ĳЩ���档
*/
//CUDA��ʹ��Eigen��
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

#define ESC_KEY 27	//�˳�����

// GPCģ��·��
#define GPC_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/GPC/sintel_small_speed/sintel_small_speed"
// ���ݼ�·��ǰ׺
#define DATA_PATH_PREFIX "E:/Paper_3DReconstruction/Dataset/"
// ��ɫ���ļ�·��ǰ׺
#define SHADER_PATH_PREFIX "E:/Paper_3DReconstruction/SparseSurfelFusion/render/shaders/"

#define CLIP_BOUNDARY 20;			// �ü��߽������

#define MAX_CAMERA_COUNT 3			// ���������(���ݼ�)������

#define MAX_RENDERED_WINDOWS 3		// ��OpenGL��Ⱦ�Ĵ����������

#define No_0_CAMERA_SERIAL_NUMBER "AY8ZA3300EW"	// 0��������к�
#define No_1_CAMERA_SERIAL_NUMBER "AY8ZA33008Y"	// 1��������к�

#define MAX_DEPTH_THRESHOLD 2500		// ���Χ2.5m
#define MIN_DEPTH_THRESHOLD 500			// ��С��Χ0.5m	

#define HIGH_FRAME_PER_SECOND		// �Ƿ�ʹ��640��400�ֱ���(���ø�֡��60FPS)

#ifdef HIGH_FRAME_PER_SECOND
#define FRAME_WIDTH  640
#define FRAME_HEIGHT 400
#define CLIP_WIDTH 600
#define CLIP_HEIGHT 360
#define FRAME_SIZE = 256000
#define MAX_SURFEL_COUNT 300000			// �����Ԫ����

#define RVM_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp16_0.8_400_640.engine"
// ONNXģ��·��(���FP16���ȵ�Engine��ʱ����Ҫ��ONNX_FP32��ת��TensorRT�ڲ���32λ��Ȩ�أ����ONNXȨ����16λ����ô��Ȩ�ص�ʱ�򣬰�����fp16���ݺϲ�һ�鵱��һ��fp32����)
#define ONNX_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp32_0.8_400_640.onnx"
#define GMFLOW_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/GMFlow/GMFlowWithDepth_400_640_fp16.engine"
#else
#define FRAME_WIDTH  1280
#define FRAME_HEIGHT 720
#define CLIP_WIDTH 1240
#define CLIP_HEIGHT 680
#define FRAME_SIZE = 921600
#define MAX_SURFEL_COUNT 100000	// �����Ԫ����
// RVMģ��·��
#define RVM_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp16_0.4_720_1280.engine"
#define GMFLOW_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/GMFlow/GMFlowWithDepth_400_640_fp16.engine"

#define ONNX_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp32_0.4_720_1280.onnx"
#endif

// ����߳���
#define MAX_THREADS 10

#define SUPER_SAMPLE_SCALE 4							// �ϲ��ó߶ȣ��ϲ���Ϊ�˱��⽫���ͼ���ڽ�������ӳ�䵽����ռ���ͬһ�����ص㴦

#define CAMERA_POSE_RIGID_ANGLE_THRESHOLD 0.8f						// ���λ��ȷ���Ƕȵ���ֵ(cos��ʾ)
#define CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD 0.03f * 0.03f	// ���λ��ȷ����Ӧ�����ֵ(����ƽ����ʾ)

#define REINITIALIZATION_TIME 30						// Canonical��ˢ��ʱ��

#define MAX_FEATURE_POINT_COUNT 2000					// �������������

#define MAX_NODE_COUNT 4096								// ���ڵ���__constant__�ڴ�ռ��64KB��GPU��__constant__�ڴ��ܹ�64KB��

#define NODE_RADIUS 0.025f								// �ڵ�֮��ľ���2.5cm
#define NODE_RADIUS_SQUARE NODE_RADIUS * NODE_RADIUS	// �ڵ�����ƽ��

#define TSDF_THRESHOLD  0.02f

//The scale of fusion map, will be accessed on device
#define d_fusion_map_scale 4

#define DRAWRECENT true

#define USE_RENDERED_RGBA_MAP_SOLVER

#define USE_MATERIALIZED_JTJ

//#define CUDA_DEBUG_SYNC_CHECK
//#define DEBUG_RUNNING_INFO

#define WITH_PCL

#define REBUILD_WITHOUT_BACKGROUND

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 400

