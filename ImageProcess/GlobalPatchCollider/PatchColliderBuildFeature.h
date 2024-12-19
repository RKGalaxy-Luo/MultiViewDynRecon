#pragma once
#include "PatchColliderCommonType.h"
namespace SparseSurfelFusion {
	
	/**
	 * \brief 根据RGB图像，构造离散余弦变换(DCT)Patch的特征提取.
	 *
	 * \param normalizedRGB 归一化的RGB图像
	 * \param centerX Patch中心坐标的x
	 * \param centerY Patch中心坐标的y
	 * \param feature 【输出】将DCT计算获得的GPC的特征系数矩阵，赋值给feature.Feature
	 */
	template<int PatchHalfSize = 10> 
	__device__ __forceinline__ void buildDCTPatchFeature(cudaTextureObject_t normalizedRGB, int centerX, int centerY, GPCPatchFeature<18>& feature);
}

// buildDCTPatchFeatureKernel核函数的实现，__forceinline__需要在头文件中完成实现
#if defined(__CUDACC__)
#include "PatchColliderBuildFeature.cuh"
#endif