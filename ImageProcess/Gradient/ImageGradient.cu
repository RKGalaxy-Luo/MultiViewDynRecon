/*****************************************************************//**
 * \file   ImageGradient.cu
 * \brief  计算图像的梯度，方法类，仅提供方法不提供内存
 * 
 * \author LUO
 * \date   March 24th 2024
 *********************************************************************/
#include "ImageGradient.h"

//			   | -3   0   3 |					| -3   -10   -3	|
// dx = 1/16 * | -10  0	 10 |		dy = 1/16 * | 0     0     0	|
//             | -3   0   3 |					| 3     10    3	|

__device__ void SparseSurfelFusion::device::computeImageGradient(const float v[8], float& dv_dx, float& dv_dy)
{
	dv_dx = 0.0625f * (-3 * v[0] + 3 * v[5] - 10 * v[1] + 10 * v[6] - 3 * v[2] + 3 * v[7]);
	dv_dy = 0.0625f * (-3 * v[0] + 3 * v[2] - 10 * v[3] + 10 * v[4] - 3 * v[5] + 3 * v[7]);
}

__global__ void SparseSurfelFusion::device::computeDensityForegroundMaskGradientKernel(cudaTextureObject_t foregroundMask, cudaTextureObject_t GrayscaleMap, unsigned int rows, unsigned int cols, cudaSurfaceObject_t foregroundMaskGradientMap, cudaSurfaceObject_t GrayscaleGradientMap)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;

	//现在梯度必须写在曲面上
	float map_value[8];
	float2 mask_gradient, density_gradient;

	//加载并计算
	map_value[0] = tex2D<float>(foregroundMask, x - 1, y - 1);
	map_value[1] = tex2D<float>(foregroundMask, x - 1, y);
	map_value[2] = tex2D<float>(foregroundMask, x - 1, y + 1);
	map_value[3] = tex2D<float>(foregroundMask, x, y - 1);
	map_value[4] = tex2D<float>(foregroundMask, x, y + 1);
	map_value[5] = tex2D<float>(foregroundMask, x + 1, y - 1);
	map_value[6] = tex2D<float>(foregroundMask, x + 1, y);
	map_value[7] = tex2D<float>(foregroundMask, x + 1, y + 1);
	computeImageGradient(map_value, mask_gradient.x, mask_gradient.y);

	map_value[0] = tex2D<float>(GrayscaleMap, x - 1, y - 1);
	map_value[1] = tex2D<float>(GrayscaleMap, x - 1, y);
	map_value[2] = tex2D<float>(GrayscaleMap, x - 1, y + 1);
	map_value[3] = tex2D<float>(GrayscaleMap, x, y - 1);
	map_value[4] = tex2D<float>(GrayscaleMap, x, y + 1);
	map_value[5] = tex2D<float>(GrayscaleMap, x + 1, y - 1);
	map_value[6] = tex2D<float>(GrayscaleMap, x + 1, y);
	map_value[7] = tex2D<float>(GrayscaleMap, x + 1, y + 1);
	computeImageGradient(map_value, density_gradient.x, density_gradient.y);

	//将值存储到surface
	surf2Dwrite(mask_gradient, foregroundMaskGradientMap, x * sizeof(float2), y);
	surf2Dwrite(density_gradient, GrayscaleGradientMap, x * sizeof(float2), y);
}

void SparseSurfelFusion::ImageGradient::computeDensityForegroundMaskGradient(cudaTextureObject_t filteredForegroundMask, cudaTextureObject_t GrayscaleMap, unsigned int rows, unsigned int cols, cudaSurfaceObject_t foregroundMaskGradientMap, cudaSurfaceObject_t GrayscaleGradientMap, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	device::computeDensityForegroundMaskGradientKernel << <grid, block, 0, stream >> > (filteredForegroundMask, GrayscaleMap, rows, cols, foregroundMaskGradientMap, GrayscaleGradientMap);
}
