/*****************************************************************//**
 * \file   CommonTypes.h
 * \brief  声明一些常见的类型，将不常见的类型重命名
 * 
 * \author LUO
 * \date   January 9th 2024
 *********************************************************************/
#pragma once
#include "GlobalConfigs.h"
#include "DeviceAPI/convenience.cuh"
#include "DeviceAPI/device_array.hpp"
#include "DeviceAPI/kernel_containers.hpp"
#include "DeviceAPI/safe_call.hpp"
/*	从 CUDA 5.5 和 Eigen 3.3 开始，可以在 CUDA 内核中使用 Eigen 的矩阵、向量和数组来实现固定大小。这在处理大量小问题时特别有用。
	默认情况下，当 Eigen 的标头包含在由 nvcc 编译的 .cu 文件中时，大多数 Eigen 的函数和方法都会以 device host 关键字作为前缀，
	从而可以从主机和设备代码调用它们。可以通过在包含任何 Eigen 的标头之前定义 EIGEN_NO_CUDA 来禁用此支持。
	当 .cu 文件仅在主机端使用 Eigen 时，这可能有助于禁用某些警告。*/
//CUDA不使用Eigen库
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif
#include <Eigen/Eigen>//包含Dense和Sparse:几乎是Eigen库的全部内容
//CUDA类型
#include <vector_functions.h>
#include <vector>
//**************************************hsg
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
typedef pcl::PointCloud<pcl::PointXYZ>    PointCloud3f;
typedef pcl::PointCloud<pcl::Normal>      PointCloudNormal;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud3fRGB;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr    PointCloud3f_Pointer;
typedef pcl::PointCloud<pcl::Normal>::Ptr      PointCloudNormal_Pointer;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud3fRGB_Pointer;

//**************************************hsg

namespace SparseSurfelFusion {
	using Matrix3f = Eigen::Matrix3f;
	using Vector3f = Eigen::Vector3f;
	using Matrix4f = Eigen::Matrix4f;
	using Vector4f = Eigen::Vector4f;
	using Matrix6f = Eigen::Matrix<float, 6, 6>;
	using Vector6f = Eigen::Matrix<float, 6, 1>;
	using MatrixXf = Eigen::MatrixXf;
	using VectorXf = Eigen::VectorXf;
	using Isometry3f = Eigen::Isometry3f;
	/* host和device访问的 gpu 容器
	*/
	//用DeviceArray代替DeviceArrayPCL
	template<typename T>
	using DeviceArray = DeviceArrayPCL<T>;
	//用DeviceArray2D代替DeviceArray2DPCL
	template<typename T>
	using DeviceArray2D = DeviceArray2DPCL<T>;

	namespace device {
		//device访问 gpu 容器的类型
		//存了Device地址的指针
		template<typename T>
		using DevicePtr = DevPtr<T>;

		//存了Device地址以及数据大小
		template<typename T>
		using PtrSize = PtrSzPCL<T>;

		//存了Device地址以及在以字节为单位的两个连续行之间的步长，可以将DeviceArray类型转成PtrStep，并在核函数中直接访问(DeviceArray在核函数中不能直接访问)
		template<typename T>
		using PtrStep = PtrStepPCL<T>;

		//存了Device地址、数据的大小以及在以字节为单位的两个连续行之间的步长
		template<typename T>
		using PtrStepSize = PtrStepSzPCL<T>;
	}

	/**
	 * 相机内参
	 */
	struct Intrinsic
	{
		/**
		 * \brief 允许在host和device上面进行Intrinsic函数构造
		 * \return
		 */
		__host__ __device__ Intrinsic()
			: principal_x(0), principal_y(0),
			focal_x(0), focal_y(0) {}

		__host__ __device__ Intrinsic(
			const float focal_x_, const float focal_y_,
			const float principal_x_, const float principal_y_
		) : principal_x(principal_x_), principal_y(principal_y_),
			focal_x(focal_x_), focal_y(focal_y_) {}

		//构造float4   [直接重载float4(),这样可以直接右赋值 -> float4 A = (Intrinsic) B]
		__host__ operator float4() {
			return make_float4(principal_x, principal_y, focal_x, focal_y);
		}

		// 相机内参
		float principal_x, principal_y;
		float focal_x, focal_y;
	};

	/**
	 * 相机内参倒数
	 */
	struct IntrinsicInverse
	{
		//默认赋初值0
		__host__ __device__ IntrinsicInverse() : principal_x(0), principal_y(0), inv_focal_x(0), inv_focal_y(0) {}

		// 相机内参
		float principal_x, principal_y; //相机内参中心点
		float inv_focal_x, inv_focal_y; //相机焦距的倒数
	};

	/**
	 * \brief 给定数组的纹理集合，其中有cudaSurfaceObject_t和cudaTextureObject_t两种类型，以及对应数据cudaArray.
	 */
	struct CudaTextureSurface {
		cudaTextureObject_t texture;	//纹理内存，可读不可写，像素点直接可以进行硬件插值
		cudaSurfaceObject_t surface;	//表面内存，可读可写，可以与OpenGL中的Resource空间进行映射或者共享，不通过CPU
		cudaArray_t cudaArray;			//创建一个Array存放数据，类型cudaArray_t，这也就是数据在GPU上实际的载体
	};

	//向上取整函数，算网格数量的，同convenience.cuh中的getGridDim()函数
	using pcl::gpu::divUp;

	/**
	 * \brief 辅助结构体记录像素.
	 */
	struct PixelCoordinate {
		unsigned int row;	// 图像的高，像素点y坐标
		unsigned int col;	// 图像的宽，像素点x坐标
		__host__ __device__ PixelCoordinate() : row(0), col(0) {}
		__host__ __device__ PixelCoordinate(const unsigned row_, const unsigned col_)
			: row(row_), col(col_) {}

		__host__ __device__ const unsigned int& x() const { return col; }
		__host__ __device__ const unsigned int& y() const { return row; }
		__host__ __device__ unsigned int& x() { return col; }
		__host__ __device__ unsigned int& y() { return row; }
	};

	/**
	 * \brief 从深度图像构建的surfel结构应该在设备上访问.
	 */
	struct DepthSurfel {
		PixelCoordinate pixelCoordinate;	// pixelCoordinate面元来自哪里
		float4 VertexAndConfidence;			// VertexAndConfidence (x, y, z)为相机帧中的位置，(w)为置信度值。
		float4 NormalAndRadius;				// NormalAndRadius (x, y, z)是归一化法线方向，w是半径
		float4 ColorAndTime;				// ColorAndTime (x)是浮点编码的RGB值;(z)为最后一次观测时间;(w)为初始化时间
		bool isMerged;						// 记录是否被融合过，true是被融合过了的面元，不再参与其他融合，false是未被融合过的面元，可以参与其他视角的融合
		short CameraID;						// 记录来自哪一个相机
	};

	struct KNNAndWeight {
		ushort4 knn;		// 临近4个点的ID
		float4 weight;		// 临近4个点的权重

		__host__ __device__ void setInvalid() {
			knn.x = knn.y = knn.z = knn.w = 0xFFFF;
			weight.x = weight.y = weight.z = weight.w = 0.0f;
		}
	};

	/**
	 * \brief 带颜色的顶点.
	 */
	struct ColorVertex {
		float3 coor;	// 三维坐标
		float3 color;	// RGB
	};

	/**
	 * \brief 用于记录跨镜匹配点.
	 */
	struct CrossViewCorrPairs {
		ushort4 PixelPairs;
		ushort2 PixelViews;
	};
	enum DatasetType {
		// 640_400_120°_1.0m
		LowRes_1000mm_calibration,      // 低分辨率，1m，标定数据集
		LowRes_1000mm_boxing_slow,      // 低分辨率，1m，[慢速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		LowRes_1000mm_boxing_fast,      // 低分辨率，1m，[快速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		LowRes_1000mm_doll_slow,        // 低分辨率，1m，[慢速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		LowRes_1000mm_doll_fast,        // 低分辨率，1m，[快速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		LowRes_1000mm_coat_slow,        // 低分辨率，1m，[慢速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		LowRes_1000mm_coat_fast,        // 低分辨率，1m，[快速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		LowRes_1000mm_claphands_slow,   // 低分辨率，1m，[慢速]两人拍手 + 两人勾手肘（主要展示多人动作）
		LowRes_1000mm_claphands_fast,   // 低分辨率，1m，[快速]两人拍手 + 两人勾手肘（主要展示多人动作）

		// 1280_720_120°_1.0m
		HighRes_1000mm_boxing_slow,     // 高分辨率，1m，[慢速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		HighRes_1000mm_boxing_fast,     // 高分辨率，1m，[快速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		HighRes_1000mm_doll_slow,       // 高分辨率，1m，[慢速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		HighRes_1000mm_doll_fast,       // 高分辨率，1m，[快速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		HighRes_1000mm_coat_slow,       // 高分辨率，1m，[慢速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		HighRes_1000mm_coat_fast,       // 高分辨率，1m，[快速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		HighRes_1000mm_claphands_slow,  // 高分辨率，1m，[慢速]两人拍手 + 两人勾手肘（主要展示多人动作）
		HighRes_1000mm_claphands_fast,  // 高分辨率，1m，[快速]两人拍手 + 两人勾手肘（主要展示多人动作）


		// 640_400_120°_1.5m
		LowRes_1500mm_calibration,      // 低分辨率，1.5m，标定数据集
		LowRes_1500mm_boxing_slow,      // 低分辨率，1.5m，[慢速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		LowRes_1500mm_boxing_fast,      // 低分辨率，1.5m，[快速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		LowRes_1500mm_doll_slow,        // 低分辨率，1.5m，[慢速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		LowRes_1500mm_doll_fast,        // 低分辨率，1.5m，[快速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		LowRes_1500mm_coat_slow,        // 低分辨率，1.5m，[慢速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		LowRes_1500mm_coat_fast,        // 低分辨率，1.5m，[快速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		LowRes_1500mm_claphands_slow,   // 低分辨率，1.5m，[慢速]两人拍手 + 两人勾手肘（主要展示多人动作）
		LowRes_1500mm_claphands_fast,   // 低分辨率，1.5m，[快速]两人拍手 + 两人勾手肘（主要展示多人动作）
		LowRes_1500mm_ChallengeTest_A,
		LowRes_1500mm_ChallengeTest_B,

		// 1280_720_120°_1.5m
		HighRes_1500mm_boxing_slow,     // 高分辨率，1.5m，[慢速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		HighRes_1500mm_boxing_fast,     // 高分辨率，1.5m，[快速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		HighRes_1500mm_doll_slow,       // 高分辨率，1.5m，[慢速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		HighRes_1500mm_doll_fast,       // 高分辨率，1.5m，[快速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		HighRes_1500mm_coat_slow,       // 高分辨率，1.5m，[慢速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		HighRes_1500mm_coat_fast,       // 高分辨率，1.5m，[快速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		HighRes_1500mm_claphands_slow,  // 高分辨率，1.5m，[慢速]两人拍手 + 两人勾手肘（主要展示多人动作）
		HighRes_1500mm_claphands_fast,  // 高分辨率，1.5m，[快速]两人拍手 + 两人勾手肘（主要展示多人动作）


		// 640_400_120°_2.0m
		LowRes_2000mm_calibration,      // 低分辨率，2.0m，标定数据集
		LowRes_2000mm_boxing_slow,      // 低分辨率，2.0m，[慢速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		LowRes_2000mm_boxing_fast,      // 低分辨率，2.0m，[快速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		LowRes_2000mm_doll_slow,        // 低分辨率，2.0m，[慢速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		LowRes_2000mm_doll_fast,        // 低分辨率，2.0m，[快速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		LowRes_2000mm_coat_slow,        // 低分辨率，2.0m，[慢速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		LowRes_2000mm_coat_fast,        // 低分辨率，2.0m，[快速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		LowRes_2000mm_claphands_slow,   // 低分辨率，2.0m，[慢速]两人拍手 + 两人勾手肘（主要展示多人动作）
		LowRes_2000mm_claphands_fast,   // 低分辨率，2.0m，[快速]两人拍手 + 两人勾手肘（主要展示多人动作）

		// 1280_720_120°_2.0m
		HighRes_2000mm_boxing_slow,     // 高分辨率，2.0m，[慢速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		HighRes_2000mm_boxing_fast,     // 高分辨率，2.0m，[快速]单人出拳 + 踢腿 + 交叉摇手（主要展示动作）
		HighRes_2000mm_doll_slow,       // 高分辨率，2.0m，[慢速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		HighRes_2000mm_doll_fast,       // 高分辨率，2.0m，[快速]揉捏玩偶 + 锤击玩偶 + 抛投玩偶 （主要展示非刚性）
		HighRes_2000mm_coat_slow,       // 高分辨率，2.0m，[慢速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		HighRes_2000mm_coat_fast,       // 高分辨率，2.0m，[快速]轻拽外套 + 脱外套 + 轻揉外套（主要展示发生拓扑变换）
		HighRes_2000mm_claphands_slow,  // 高分辨率，2.0m，[慢速]两人拍手 + 两人勾手肘（主要展示多人动作）
		HighRes_2000mm_claphands_fast,  // 高分辨率，2.0m，[快速]两人拍手 + 两人勾手肘（主要展示多人动作）
		LowRes_2000mm_ChallengeTest_A,
		LowRes_2000mm_ChallengeTest_B,
		LowRes_2000mm_ChallengeTest_C,
	};

	void DatasetSwitch(DatasetType type, std::string& res, std::string& dis, std::string& speed, std::string& action);
}