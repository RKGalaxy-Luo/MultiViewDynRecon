#pragma once
/*****************************************************************//**
 * \file   CrossViewMatchingInterpolation.h
 * \brief  针对跨镜匹配点进行边缘插值
 * 
 * 
 * \author LUO
 * \date   December 2024
 *********************************************************************/
#include "CrossViewEdgeCorrespondence.h"

#include <math/DualQuaternion/DualQuaternion.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>

#include <visualization/Visualizer.h>

namespace SparseSurfelFusion {
	namespace device {
		struct CrossViewInterpolateInput {
			cudaTextureObject_t VertexMap[MAX_CAMERA_COUNT];				// vertexMap
			cudaTextureObject_t NormalMap[MAX_CAMERA_COUNT];				// normalMap数组
			cudaTextureObject_t ColorMap[MAX_CAMERA_COUNT];					// colorMap数组
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];						// 视角位姿
			mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];					// 视角位姿
			Intrinsic intrinsic[MAX_CAMERA_COUNT];							// 相机内参
		};

		struct CrossViewInterpolateOutput {
			PtrStepSize<float4> interVertexMap[MAX_CAMERA_COUNT];			// 插值点的vertexMap
			PtrStepSize<float4> interNormalMap[MAX_CAMERA_COUNT];			// 插值点的NormalMap
			PtrStepSize<float4> interColorMap[MAX_CAMERA_COUNT];			// 插值点的ColorMap

			PtrStepSize<unsigned int> mutexFlag[MAX_CAMERA_COUNT];		// 防止多个插值点投影到同一个像素点造成访存错误
			PtrStepSize<unsigned char> markInterValue[MAX_CAMERA_COUNT];	// 标记有效的插值点位置
		};

		struct CorrectTextureIO {
			// 插值数据
			DeviceArrayView2D<float4> interVertexMap[MAX_CAMERA_COUNT];			// 插值点的vertexMap
			DeviceArrayView2D<float4> interNormalMap[MAX_CAMERA_COUNT];			// 插值点的NormalMap
			DeviceArrayView2D<float4> interColorMap[MAX_CAMERA_COUNT];			// 插值点的ColorMap

			DeviceArrayView2D<unsigned int> mutexFlag[MAX_CAMERA_COUNT];		// 防止多个插值点投影到同一个像素点造成访存错误
			DeviceArrayView2D<unsigned char> markInterValue[MAX_CAMERA_COUNT];	// 标记有效的插值点位置

			// 观测数据
			cudaTextureObject_t VertexTextureMap[MAX_CAMERA_COUNT];			// 当前帧观测只读的VertexMap
			cudaTextureObject_t NormalTextureMap[MAX_CAMERA_COUNT];			// 当前帧观测只读的NormalMap
			cudaTextureObject_t ColorTextureMap[MAX_CAMERA_COUNT];			// 当前帧观测只读的ColorMap

			cudaSurfaceObject_t VertexSurfaceMap[MAX_CAMERA_COUNT];			// 当前帧观测可写入的VertexMap
			cudaSurfaceObject_t NormalSurfaceMap[MAX_CAMERA_COUNT]; 		// 当前帧观测可写入的NormalMap
			cudaSurfaceObject_t ColorSurfaceMap[MAX_CAMERA_COUNT];			// 当前帧观测可写入的ColorMap
		};

		/**
		 * \brief 沙勒螺旋运动.
		 */
		__device__ mat34 ScrewInterpolationMat(DualQuaternion dq, float ratio);

		/**
		 * \brief 跨镜面元插值.
		 */
		__global__ void SurfelsInterpolationKernel(CrossViewInterpolateInput input, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, const unsigned int Cols, const unsigned int Rows, const unsigned int pairsNum, const float disThreshold, CrossViewInterpolateOutput output);
	
		/**
		 * \brief 校正观测的Texture.
		 */
		__global__ void CorrectObservedTextureKernel(CorrectTextureIO io, const unsigned int Cols, const unsigned int Rows, const unsigned int CameraNum);
	}

	class CrossViewMatchingInterpolation
	{
	public:
		using Ptr = std::shared_ptr<CrossViewMatchingInterpolation>;

		CrossViewMatchingInterpolation(Intrinsic* intrinsicArray, mat34* initialPoseArray);

		~CrossViewMatchingInterpolation();

		struct CrossViewInterpolationInput {
			cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];				// vertexMap数组
			cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];				// normalMap数组
			cudaTextureObject_t colorMap[MAX_CAMERA_COUNT];					// colorMap数组
		};



	private:
		const unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int ImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int ImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageSize = FRAME_HEIGHT * FRAME_WIDTH;
		const unsigned int clipedImageSize = ImageRowsCliped * ImageColsCliped;

		DeviceArray2D<float4> interVertexMap[MAX_CAMERA_COUNT];			// 插值点的vertexMap
		DeviceArray2D<float4> interNormalMap[MAX_CAMERA_COUNT];			// 插值点的NormalMap
		DeviceArray2D<float4> interColorMap [MAX_CAMERA_COUNT];			// 插值点的ColorMap

		DeviceArray2D<unsigned int> mutexFlag[MAX_CAMERA_COUNT];		// 防止多个插值点投影到同一个像素点造成访存错误
		DeviceArray2D<unsigned char> markInterValue[MAX_CAMERA_COUNT];	// 标记有效的插值点位置

		device::CrossViewInterpolateInput interpolationInput;			// 计算插值点的输入
		device::CrossViewInterpolateOutput interpolationOutput;			// 插值点对应输出
		device::CorrectTextureIO correctedIO;							// 修改观测帧的IO

		DeviceArrayView<CrossViewCorrPairs> crossCorrPairs;				// 传入的跨镜匹配点
		
	public:
		/**
		 * \brief 跨镜插值输入.
		 * 
		 * \param input 输入
		 * \param crossPairs 跨镜匹配点
		 */
		void SetCrossViewInterpolationInput(const CrossViewInterpolationInput& input, DeviceArrayView<CrossViewCorrPairs> crossPairs);

		/**
		 * \brief 跨镜匹配插值Surfels.
		 * 
		 * \param stream cuda流
		 */
		void CrossViewInterpolateSurfels(cudaStream_t stream = 0);

		/**
		 * \brief 校正VertexMap, NormalMap, ColorMap, 将插值点融入进去.
		 * 
		 * \param VertexMap 顶点Map
		 * \param NormalMap 法线Map
		 * \param ColorMap 颜色Map
		 * \param stream cuda流
		 */
		void CorrectVertexNormalColorTexture(CudaTextureSurface* VertexMap, CudaTextureSurface* NormalMap, CudaTextureSurface* ColorMap, cudaStream_t stream = 0);
	
		/**
		 * \brief 获得某个平面的插值VertexMap.
		 * 
		 * \param CameraID 相机视角ID
		 * \return 平面的插值VertexMap
		 */
		DeviceArrayView2D<float4> GetInterpolatedVertexMap(const unsigned int CameraID) { return correctedIO.interVertexMap[CameraID]; }
		
		/**
		 * \brief 获得某个平面的插值NormalMap.
		 * 
		 * \param CameraID 相机视角ID
		 * \return 平面的插值NormalMap
		 */
		DeviceArrayView2D<float4> GetInterpolatedNormalMap(const unsigned int CameraID) { return correctedIO.interNormalMap[CameraID]; }

		/**
		 * \brief 获得某个平面的插值ColorMap.
		 * 
		 * \param CameraID 相机视角ID
		 * \return 平面的插值ColorMap
		 */
		DeviceArrayView2D<float4> GetInterpolatedColorMap(const unsigned int CameraID) { return correctedIO.interColorMap[CameraID]; }

		/**
		 * \brief 获得有效的插值标记Map.
		 * 
		 * \param CameraID 相机视角ID
		 * \return 有效的插值标记Map
		 */
		DeviceArrayView2D<uchar> GetValidInterpolatedMarkMap(const unsigned int CameraID) { return correctedIO.markInterValue[CameraID]; }
	
	private:
		/**
		 * \brief 设置初始值.
		 * 
		 * \param stream cuda流
		 */
		void SetInitialValue(cudaStream_t stream = 0);
	};
}


