#include <base/ColorTypeTransfer.h>
#include "PatchColliderBuildFeature.h"

template<int PatchHalfSize = 10>
__device__ __forceinline__ void SparseSurfelFusion::buildDCTPatchFeature(cudaTextureObject_t normalizedRGB, int centerX, int centerY, GPCPatchFeature<18>& feature)
{
	const float pi = 3.1415926f;
	const int left_x = centerX - PatchHalfSize;	// 当前Patch的左上角x坐标
	const int top_y = centerY - PatchHalfSize;	// 当前Patch的左上角y坐标

	// YCbCr（也称为YCC）是一种颜色编码系统，用于数字图像和视频中的色彩表示。它是一种亮度 - 色度编码
	// 由于人眼对亮度变化更敏感，而对色度变化不太敏感，使用亮度-色度编码可以在保持图像质量的同时减少数据传输和存储的需求

	// 将Patch加载到局部内存中
	float3 patch_ycrcb[2 * PatchHalfSize][2 * PatchHalfSize];
	for (int y = top_y; y < top_y + 2 * PatchHalfSize; y++) {
		for (int x = left_x; x < left_x + 2 * PatchHalfSize; x++)
		{
			// 读取纹理
			const float4 rgba = tex2D<float4>(normalizedRGB, x, y);

			// 转格式
			float3 ycrcb;
			normalized_rgba2ycrcb(rgba, ycrcb);
			ycrcb.x *= 255;
			ycrcb.y *= 255;
			ycrcb.z *= 255;

			// 注意y索引是在第一个
			patch_ycrcb[y - top_y][x - left_x] = ycrcb;
		}
	}

	// DCT迭代循环 4×4系数矩阵feature.Feature(一维形式存储)
	for (int n0 = 0; n0 < 4; n0++) {
		for (int n1 = 0; n1 < 4; n1++) {
			float dct_sum = 0.0f;
			for (int y = 0; y < 2 * PatchHalfSize; y++) {
				for (int x = 0; x < 2 * PatchHalfSize; x++) {
					// 读取图像纹理数据，只处理明度分量x
					const float3 ycrcb = patch_ycrcb[y][x];
					dct_sum += ycrcb.x 
						* cosf(pi * (x + 0.5f) * n0 / (2 * PatchHalfSize)) 
						* cosf(pi * (y + 0.5f) * n1 / (2 * PatchHalfSize));
				}
			}
			// 保存Patch
			feature.Feature[n0 * 4 + n1] = dct_sum / float(PatchHalfSize);
		}
	}

	// 调整描述子的尺度
	for (int k = 0; k < 4; k++) {
		feature.Feature[k] *= 0.7071067811865475f;		// 对前4个描述子(0,1,2,3)，乘以1/√2
		feature.Feature[k * 4] *= 0.7071067811865475f;	// 对0，4，8，12号描述子进行了额外尺度调整，因为在DCT变换中，这些位置上的系数具有较高的能量，需要额外进行尺度调整以保持平衡
	}


	// 最后两个描述符：计算色度y和色度z的平均值
	float cr_sum = 0.0f;
	float cb_sum = 0.0f;
	for (int y = 0; y < 2 * PatchHalfSize; y++) {
		for (int x = 0; x < 2 * PatchHalfSize; x++) {
			// 读取图像纹理数据
			const float3 ycrcb = patch_ycrcb[y][x];
			cr_sum += ycrcb.y;
			cb_sum += ycrcb.z;
		}
	}
	feature.Feature[16] = cr_sum / (2 * PatchHalfSize);
	feature.Feature[17] = cb_sum / (2 * PatchHalfSize);

}
