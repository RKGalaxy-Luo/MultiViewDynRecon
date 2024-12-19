/*****************************************************************//**
 * \file   ImageGradient.h
 * \brief  ����ͼ����ݶȣ������࣬���ṩ�������ṩ�ڴ�
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
		 * \brief ����ͼ���ݶ�.
		 * 
		 *				 v0, v3, v5
		 * \param v[8] = v1,   , v6
		 *				 v2, v4, v7
		 * \param dv_dx x�����ϵ��ݶ�
		 * \param dv_dy y�����ϵ��ݶ�
		 */
		__device__ void computeImageGradient(const float v[8], float& dv_dx, float& dv_dy);

		/**
		 * \brief �����˹�ǰ���ɰ���ܶ�(�Ҷ�)ͼ��ͼ���ݶ�
		 * \param foregroundMask �˹�ǰ���ɰ� float1����
		 * \param GrayscaleMap �ܶ�ͼ float1����
		 * \param rows ����ͼƬ��row
		 * \param cols ����ͼƬ��col
		 * \param foregroundMaskGradientMap ǰ���ɰ��ݶ�ͼ float2����
		 * \param GrayscaleGradientMap �ܶ��ݶ�ͼ float2����
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
		 * \brief �����˹�ǰ���ɰ���ܶ�(�Ҷ�)ͼ��ͼ���ݶ�
		 * \param foregroundMask �˹�ǰ���ɰ� float1����
		 * \param GrayscaleMap �ܶ�ͼ float1����
		 * \param rows ����ͼƬ��row
		 * \param cols ����ͼƬ��col
		 * \param foregroundMaskGradientMap ǰ���ɰ��ݶ�ͼ float2����
		 * \param GrayscaleGradientMap �ܶ��ݶ�ͼ float2����
		 * \param stream cuda��ID
		 */
		void computeDensityForegroundMaskGradient(cudaTextureObject_t foregroundMask, cudaTextureObject_t GrayscaleMap, unsigned int rows, unsigned int cols, cudaSurfaceObject_t foregroundMaskGradientMap, cudaSurfaceObject_t GrayscaleGradientMap, cudaStream_t stream = 0);
	};
}


