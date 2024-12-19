/*****************************************************************//**
 * \file   ColorTypeTransfer.h
 * \brief  重要供核函数调用，从而转换图像的类型
 * 
 * \author LUO
 * \date   January 29th 2024
 *********************************************************************/
#pragma once
#include <math/VectorUtils.h>

namespace SparseSurfelFusion {
	/**
	 * \brief 将像素点从RGBA类型转到YCrCb类型.
	 * 
	 * \param rgba 输入RGBA类型像素点
	 * \param ycrcb 输出YCrCb类型像素点
	 */
	__host__ __device__ __forceinline__ void normalized_rgba2ycrcb(const float4& rgba, float3& ycrcb) {
		ycrcb.x = 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
		ycrcb.y = -0.168736f * rgba.x - 0.331264f * rgba.y + 0.5f * rgba.z;
		ycrcb.z = 0.5f * rgba.x - 0.418688f * rgba.y - 0.091312f * rgba.z;
	}
	/**
	 * \brief 将像素点从RGB类型转到YCrCb类型.
	 * 
	 * \param rgb 输入RGBA类型像素点
	 * \param ycrcb 输出YCrCb类型像素点
	 */
	__host__ __device__ __forceinline__ void normalized_rgb2ycrcb(const float3& rgb, float3& ycrcb) {
		ycrcb.x = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
		ycrcb.y = -0.168736f * rgb.x - 0.331264f * rgb.y + 0.5f * rgb.z;
		ycrcb.z = 0.5f * rgb.x - 0.418688f * rgb.y - 0.091312f * rgb.z;
	}

	/**
	 * \brief 将像素点从RGBA类型转到灰度类型.
	 * 
	 * \param rgba 输入RGBA类型像素点
	 * \return 输出当前RGBA值对应的灰度像素的值
	 */
	__host__ __device__ __forceinline__ float rgba2density(const float4& rgba) {
		return 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
	}

	/**
	 * \brief 将像素点从RGB类型转到灰度类型.
	 * 
	 * \param rgb 输入RGB类型像素点
	 * \return 输出当前RGB值对应的灰度像素的值
	 */
	__host__ __device__ __forceinline__ float rgb2density(const float3& rgb) {
		return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
	}
	__host__ __device__ __forceinline__ float rgb2density(const float4& rgb) {
		return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
	}

	/**
	 * \brief 计算两个RGB图像对应像素的差分.
	 * 
	 * \param rgb_0 图像0
	 * \param rgb_1 图像1
	 * \return 对应位置像素的差分
	 */
	__host__ __device__ __forceinline__ unsigned char rgb_diff_abs(const uchar3& rgb_0, const uchar3& rgb_1) {
		return abs(rgb_0.x - rgb_1.x) + abs(rgb_0.y - rgb_1.y) + abs(rgb_0.z - rgb_1.z);
	}
}
