/*****************************************************************//**
 * \file   ColorTypeTransfer.h
 * \brief  ��Ҫ���˺������ã��Ӷ�ת��ͼ�������
 * 
 * \author LUO
 * \date   January 29th 2024
 *********************************************************************/
#pragma once
#include <math/VectorUtils.h>

namespace SparseSurfelFusion {
	/**
	 * \brief �����ص��RGBA����ת��YCrCb����.
	 * 
	 * \param rgba ����RGBA�������ص�
	 * \param ycrcb ���YCrCb�������ص�
	 */
	__host__ __device__ __forceinline__ void normalized_rgba2ycrcb(const float4& rgba, float3& ycrcb) {
		ycrcb.x = 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
		ycrcb.y = -0.168736f * rgba.x - 0.331264f * rgba.y + 0.5f * rgba.z;
		ycrcb.z = 0.5f * rgba.x - 0.418688f * rgba.y - 0.091312f * rgba.z;
	}
	/**
	 * \brief �����ص��RGB����ת��YCrCb����.
	 * 
	 * \param rgb ����RGBA�������ص�
	 * \param ycrcb ���YCrCb�������ص�
	 */
	__host__ __device__ __forceinline__ void normalized_rgb2ycrcb(const float3& rgb, float3& ycrcb) {
		ycrcb.x = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
		ycrcb.y = -0.168736f * rgb.x - 0.331264f * rgb.y + 0.5f * rgb.z;
		ycrcb.z = 0.5f * rgb.x - 0.418688f * rgb.y - 0.091312f * rgb.z;
	}

	/**
	 * \brief �����ص��RGBA����ת���Ҷ�����.
	 * 
	 * \param rgba ����RGBA�������ص�
	 * \return �����ǰRGBAֵ��Ӧ�ĻҶ����ص�ֵ
	 */
	__host__ __device__ __forceinline__ float rgba2density(const float4& rgba) {
		return 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
	}

	/**
	 * \brief �����ص��RGB����ת���Ҷ�����.
	 * 
	 * \param rgb ����RGB�������ص�
	 * \return �����ǰRGBֵ��Ӧ�ĻҶ����ص�ֵ
	 */
	__host__ __device__ __forceinline__ float rgb2density(const float3& rgb) {
		return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
	}
	__host__ __device__ __forceinline__ float rgb2density(const float4& rgb) {
		return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
	}

	/**
	 * \brief ��������RGBͼ���Ӧ���صĲ��.
	 * 
	 * \param rgb_0 ͼ��0
	 * \param rgb_1 ͼ��1
	 * \return ��Ӧλ�����صĲ��
	 */
	__host__ __device__ __forceinline__ unsigned char rgb_diff_abs(const uchar3& rgb_0, const uchar3& rgb_1) {
		return abs(rgb_0.x - rgb_1.x) + abs(rgb_0.y - rgb_1.y) + abs(rgb_0.z - rgb_1.z);
	}
}
