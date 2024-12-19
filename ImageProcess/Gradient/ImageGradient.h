/*****************************************************************//**
 * \file   ImageGradient.h
 * \brief  计算图像的梯度，方法类，仅提供方法不提供内存
 * 
 * \author LUO
 * \date   March 24th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <device_launch_parameters.h>
namespace SparseSurfelFusion {

	namespace device {

		/**
		 * \brief 计算图像梯度.
		 * 
		 *				 v0, v3, v5
		 * \param v[8] = v1,   , v6
		 *				 v2, v4, v7
		 * \param dv_dx x方向上的梯度
		 * \param dv_dy y方向上的梯度
		 */
		__device__ void computeImageGradient(const float v[8], float& dv_dx, float& dv_dy);

		/**
		 * \brief 计算滤光前景蒙版和密度(灰度)图的图像梯度
		 * \param foregroundMask 滤光前景蒙版 float1纹理
		 * \param GrayscaleMap 密度图 float1纹理
		 * \param rows 所有图片的row
		 * \param cols 所有图片的col
		 * \param foregroundMaskGradientMap 前景蒙版梯度图 float2纹理
		 * \param GrayscaleGradientMap 密度梯度图 float2纹理
		 */
		__global__ void computeDensityForegroundMaskGradientKernel(cudaTextureObject_t foregroundMask, cudaTextureObject_t GrayscaleMap, unsigned int rows, unsigned int cols, cudaSurfaceObject_t foregroundMaskGradientMap, cudaSurfaceObject_t GrayscaleGradientMap);
	}
	class ImageGradient
	{
	public:
		using Ptr = std::shared_ptr<ImageGradient>;
		ImageGradient() = default;
		~ImageGradient() = default;

		/**
		 * \brief 计算滤光前景蒙版和密度(灰度)图的图像梯度
		 * \param foregroundMask 滤光前景蒙版 float1纹理
		 * \param GrayscaleMap 密度图 float1纹理
		 * \param rows 所有图片的row
		 * \param cols 所有图片的col
		 * \param foregroundMaskGradientMap 前景蒙版梯度图 float2纹理
		 * \param GrayscaleGradientMap 密度梯度图 float2纹理
		 * \param stream cuda流ID
		 */
		void computeDensityForegroundMaskGradient(cudaTextureObject_t foregroundMask, cudaTextureObject_t GrayscaleMap, unsigned int rows, unsigned int cols, cudaSurfaceObject_t foregroundMaskGradientMap, cudaSurfaceObject_t GrayscaleGradientMap, cudaStream_t stream = 0);
	};
}


