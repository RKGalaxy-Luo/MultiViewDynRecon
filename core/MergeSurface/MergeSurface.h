/*****************************************************************//**
 * \file   MergeSurface.h
 * \brief  使用插值虚拟相机的方式融合两个曲面
 * 
 * \author LUOJIAXUAN
 * \date   June 14th 2024
 *********************************************************************/
#pragma once

#include <chrono>

#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <math/MatUtils.h>
#include <base/EncodeUtils.h>
#include <base/Constants.h>
#include <base/CommonTypes.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
namespace SparseSurfelFusion {

	// 记录相机内参外参以及ID
	struct CameraParameters {
		Intrinsic intrinsic;	// 相机内参
		mat34 SE3;				// 相机相对于基准相机的位姿
		short ID;				// 相机ID
	};

	namespace device {
		enum {
			ReduceThreadsPerBlock = 128
		};

		/**
		 * \brief 计算将两个面投影到插值面重叠像素的数量.
		 * 
		 * \param Camera_1_SE3 相机1相对于基准相机的位姿
		 * \param Camera_1_intrinsic 相机1的内参的Inverse
		 * \param Camera_2_SE3 相机2相对于基准相机的位姿的
		 * \param Camera_2_intrinsic 相机2的内参Inverse
		 * \param InterprolateCameraSE3 插值相机相对于基准相机的位姿
		 * \param InterprolateCameraIntrinsic 插值相机的内参
		 * \param clipedCols 插值相机像素平面的宽
		 * \param clipedRows 插值相机像素平面的高
		 * \param totalSurfelNum 两个相机获得的深度面元的总数
		 * \param Camera_1_DepthSurfels 相机1的相机坐标系的面元
		 * \param Camera_2_DepthSurfels 相机2的相机坐标系的面元
		 * \param OverlappingIndex 获得当前重叠面元被统计时的顺序
		 * \param overlappingSurfelsCountMap 重叠像素统计一维数组
		 */
		__global__ void CountOverlappingSurfelsKernel(const mat34 Camera_1_SE3, const Intrinsic Camera_1_intrinsic, const mat34 Camera_2_SE3, const Intrinsic Camera_2_intrinsic, const mat34 InterprolateCameraSE3, const Intrinsic InterprolateCameraIntrinsic, const unsigned int clipedCols, const unsigned int clipedRows, const unsigned int Camera_1_SurfelsNums, const unsigned int totalSurfelNum, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, unsigned short* OverlappingIndex, unsigned int* overlappingSurfelsCountMap, ushort2* surfelProjectedPixelPos);

		/**
		 * \brief 规约求最大的重叠点的数量.
		 * 
		 * \param overlappingSurfelsCountMap 重叠面元的数量数组
		 * \param clipedImageSize 剪裁图像大小
		 * \param MaxCountData 最大的数量记录
		 */
		__global__ void reduceMaxOverlappingSurfelCountKernel(const unsigned int* overlappingSurfelsCountMap, const unsigned int clipedImageSize, unsigned int* MaxCountData);

		/**
		 * \brief 找到重叠点的最大数量.
		 * 
		 * \param maxCountDataHost 传入每个gird中最大的值
		 * \param gridNum grid的数量
		 * \param MaxValue 最大值
		 */
		__host__ void findMaxValueOfOverlappingCount(const unsigned int* maxCountDataHost, const unsigned int gridNum, unsigned int& MaxValue);

		/**
		 * \brief 收集在像素处重叠的surfel.
		 */
		__global__ void CollectOverlappingSurfelInPixel(const DepthSurfel* Camera_1_DepthSurfels, const DepthSurfel* Camera_2_DepthSurfels, const unsigned int* overlappingSurfelsCountMap, const unsigned short* OverlappingOrderMap, const ushort2* surfelProjectedPixelPos, const unsigned int Camera_1_SurfelCount, const unsigned int totalSurfels, const unsigned int clipedCols, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, unsigned int* SurfelIndexArray);


		/**
		 * \brief 根据TSDF重新估算重叠面的Surfel位置.
		 */
		__global__ void CalculateOverlappingSurfelTSDF(const CameraParameters Camera_1_Para, const CameraParameters Camera_2_Para, const CameraParameters Camera_inter_Para, const float maxMergeSquaredDistance, const unsigned int* overlappingSurfelsCountMap, const unsigned int* SurfelIndexArray, const unsigned int Camera_1_SurfelNum, const unsigned int clipedImageSize, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, float* MergeSurfelTSDFValue, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, bool* markValidMergedSurfel);

		/**
		 * \brief 计算两点距离的平方的值.
		 */
		__forceinline__ __device__ float CalculateSquaredDistance(const float4& v1, const float4& v2);

		/**
		 * \brief 计算TSDF的值.
		 */
		__forceinline__ __device__ float CalculateTSDFValue(float zeroPoint, float otherPoint);

		/**
		 * \brief 解码RGB编码.
		 */
		__forceinline__ __device__ void DecodeFloatRGB(const float RGBCode, uchar3& color);

		/**
		 * \brief 归一化法线.
		 */
		__forceinline__ __device__ float3 NormalizeNormal(float& nx, float& ny, float& nz);

		/**
		 * \brief 标记没有参与融合的点，并将这些点转换到Canonical域中.
		 */
		__global__ void MarkAndConvertValidNotMergedSurfeltoCanonical(const DepthSurfel* SurfelsArray, const CameraParameters cameraPara, const unsigned int SurfelsNums, DepthSurfel* SurfelsCanonical, bool* markValidNotMergedFlag);

	}
	class MergeSurface
	{
	public:
		/**
			* \brief 传入需要融合的相机曲面，以相机0为主相机，以相机1为次相机.
			* 
			* \param CameraID_0 相机0
			* \param CameraID_1 相机1
			* \param intrinsics 所有相机剪裁后的内参矩阵
			*/
		MergeSurface(const Intrinsic* intrinsics);
		~MergeSurface();
		using Ptr = std::shared_ptr<MergeSurface>;

		void MergeAllSurfaces(DeviceBufferArray<DepthSurfel>* depthSurfelsView);

		/**
		 * \brief 获得融合后的面元，当前所有面元均在Canonical域.
		 * 
		 * \return 融合后的Canonical域面元【只读】
		 */
		DeviceArrayView<DepthSurfel> GetMergedSurfelsView() { return MergedDepthSurfels.ArrayView(); }

		/**
		 * \brief 获得融合后的面元，当前所有面元均在Canonical域.
		 *
		 * \return 融合后的Canonical域面元【可读写】
		 */
		DeviceArray<DepthSurfel> GetMergedSurfelArray() { return MergedDepthSurfels.Array(); }

	private:
		const unsigned int CamerasCount = MAX_CAMERA_COUNT;				// 相机数量
		const unsigned int ClipedImageSize = CLIP_HEIGHT * CLIP_WIDTH;	// 剪裁图片大小
		const unsigned int clipedCols = CLIP_WIDTH;
		const unsigned int clipedRows = CLIP_HEIGHT;
		DeviceBufferArray<DepthSurfel> MergedDepthSurfels;				// 融合后的深度面元，面元在Canonical域
		CameraParameters CameraParameter[2 * MAX_CAMERA_COUNT];			// 记录相机内参外参以及ID，真实相机【偶数】，插值相机【奇数】
		DeviceBufferArray<unsigned int> OverlappingSurfelsCountMap;		// 记录重叠面的数量(使用short貌似会影响原子操作的性能)
		DeviceBufferArray<unsigned short> OverlappingOrderMap;			// 记录两个视角的面元计算重叠时的顺序[0, 65535]
		DeviceBufferArray<ushort2> surfelProjectedPixelPos;				// 记录当前面元被投影到了插值相机的哪个像素点
		cudaStream_t stream;											// cuda流


		/**
		 * \brief 计算参与TSDF融合，并将融合后的面元存入MergedDepthSurfels.
		 * 
		 * \param Camera_1_Para 相机1内外参
		 * \param Camera_2_Para 相机2内外参
		 * \param Camera_Inter_Para 插值相机内外参
		 * \param Camera_1_DepthSurfels 来自相机1的深度面元
		 * \param Camera_2_DepthSurfels 来自相机1的深度面元
		 * \param stream cuda流
		 */
		void CalculateTSDFMergedSurfels(const CameraParameters Camera_1_Para, const CameraParameters Camera_2_Para, const CameraParameters Camera_Inter_Para, DeviceArray<DepthSurfel> Camera_1_DepthSurfels, DeviceArray<DepthSurfel> Camera_2_DepthSurfels, cudaStream_t stream = 0);

		/**
		 * \brief 收集没有参与融合的面元.
		 * 
		 * \param depthSurfels 深度面元
		 * \param stream cuda流
		 */
		void CollectNotMergedSurfels(DeviceBufferArray<DepthSurfel>* depthSurfels, cudaStream_t stream = 0);
	};
	
}


