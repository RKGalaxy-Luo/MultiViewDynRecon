/*****************************************************************//**
 * \file   GlobalConfigs.h
 * \brief  全局参数
 * 
 * \author LUO
 * \date   January 6th 2024
 *********************************************************************/
#pragma once


/*	
 * 从 CUDA 5.5 和 Eigen 3.3 开始，可以在 CUDA 内核中使用 Eigen 的矩阵、向量和数组来实现固定大小。这在处理大量小问题时特别有用。
 * 默认情况下，当 Eigen 的标头包含在由 nvcc 编译的 .cu 文件中时，大多数 Eigen 的函数和方法都会以 device host 关键字作为前缀，
 * 从而可以从主机和设备代码调用它们。可以通过在包含任何 Eigen 的标头之前定义 EIGEN_NO_CUDA 来禁用此支持。
 * 当 .cu 文件仅在主机端使用 Eigen 时，这可能有助于禁用某些警告。
*/
//CUDA不使用Eigen库
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

#define ESC_KEY 27	//退出按键

// GPC模型路径
#define GPC_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/GPC/sintel_small_speed/sintel_small_speed"
// 数据集路径前缀
#define DATA_PATH_PREFIX "E:/Paper_3DReconstruction/Dataset/"
// 着色器文件路径前缀
#define SHADER_PATH_PREFIX "E:/Paper_3DReconstruction/SparseSurfelFusion/render/shaders/"

#define CLIP_BOUNDARY 20;			// 裁剪边界的像素

#define MAX_CAMERA_COUNT 3			// 最大接入相机(数据集)的数量

#define MAX_RENDERED_WINDOWS 3		// 用OpenGL渲染的窗口最大数量

#define No_0_CAMERA_SERIAL_NUMBER "AY8ZA3300EW"	// 0号相机序列号
#define No_1_CAMERA_SERIAL_NUMBER "AY8ZA33008Y"	// 1号相机序列号

#define MAX_DEPTH_THRESHOLD 2500		// 最大范围2.5m
#define MIN_DEPTH_THRESHOLD 500			// 最小范围0.5m	

#define HIGH_FRAME_PER_SECOND		// 是否使用640×400分辨率(采用高帧率60FPS)

#ifdef HIGH_FRAME_PER_SECOND
#define FRAME_WIDTH  640
#define FRAME_HEIGHT 400
#define CLIP_WIDTH 600
#define CLIP_HEIGHT 360
#define FRAME_SIZE = 256000
#define MAX_SURFEL_COUNT 300000			// 最大面元个数

#define RVM_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp16_0.8_400_640.engine"
// ONNX模型路径(获得FP16精度的Engine的时候需要用ONNX_FP32来转，TensorRT内部以32位读权重，如果ONNX权重是16位，那么读权重的时候，把两个fp16数据合并一块当成一个fp32读了)
#define ONNX_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp32_0.8_400_640.onnx"
#define GMFLOW_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/GMFlow/GMFlowWithDepth_400_640_fp16.engine"
#else
#define FRAME_WIDTH  1280
#define FRAME_HEIGHT 720
#define CLIP_WIDTH 1240
#define CLIP_HEIGHT 680
#define FRAME_SIZE = 921600
#define MAX_SURFEL_COUNT 100000	// 最大面元个数
// RVM模型路径
#define RVM_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp16_0.4_720_1280.engine"
#define GMFLOW_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/GMFlow/GMFlowWithDepth_400_640_fp16.engine"

#define ONNX_MODEL_PATH "E:/Paper_3DReconstruction/AlgorithmResource/RVM/rvm_resnet50_fp32_0.4_720_1280.onnx"
#endif

// 最大线程数
#define MAX_THREADS 10

#define SUPER_SAMPLE_SCALE 4							// 上采用尺度：上采用为了避免将深度图中邻近的像素映射到相机空间中同一个像素点处

#define CAMERA_POSE_RIGID_ANGLE_THRESHOLD 0.8f						// 相机位姿确定角度的阈值(cos表示)
#define CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD 0.03f * 0.03f	// 相机位姿确定对应点的阈值(距离平方表示)

#define REINITIALIZATION_TIME 30						// Canonical域刷新时间

#define MAX_FEATURE_POINT_COUNT 2000					// 特征点采样数量

#define MAX_NODE_COUNT 4096								// 最大节点数__constant__内存占用64KB【GPU中__constant__内存总共64KB】

#define NODE_RADIUS 0.025f								// 节点之间的距离2.5cm
#define NODE_RADIUS_SQUARE NODE_RADIUS * NODE_RADIUS	// 节点距离的平方

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

