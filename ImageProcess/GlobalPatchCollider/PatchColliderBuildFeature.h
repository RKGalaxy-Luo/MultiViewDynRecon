#pragma once
#include "PatchColliderCommonType.h"
namespace SparseSurfelFusion {
	
	/**
	 * \brief ����RGBͼ�񣬹�����ɢ���ұ任(DCT)Patch��������ȡ.
	 *
	 * \param normalizedRGB ��һ����RGBͼ��
	 * \param centerX Patch���������x
	 * \param centerY Patch���������y
	 * \param feature ���������DCT�����õ�GPC������ϵ�����󣬸�ֵ��feature.Feature
	 */
	template<int PatchHalfSize = 10> 
	__device__ __forceinline__ void buildDCTPatchFeature(cudaTextureObject_t normalizedRGB, int centerX, int centerY, GPCPatchFeature<18>& feature);
}

// buildDCTPatchFeatureKernel�˺�����ʵ�֣�__forceinline__��Ҫ��ͷ�ļ������ʵ��
#if defined(__CUDACC__)
#include "PatchColliderBuildFeature.cuh"
#endif