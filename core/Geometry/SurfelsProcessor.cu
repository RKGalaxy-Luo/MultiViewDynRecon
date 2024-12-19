/*****************************************************************//**
 * \file   SurfelsProcessor.h
 * \brief  用于对于面元的一些基本处理，融合、下采样、提取分量等
 *
 * \author LUO
 * \date   March 7th 2024
 *********************************************************************/
#include "SurfelsProcessor.h"

__global__ void SparseSurfelFusion::device::MergeDenseSurfelToCanonicalFieldKernel(
	DeviceArrayHandle<DepthSurfel> mergedSurfels, 
	DeviceArrayView<DepthSurfel> depthSurfel, 
	mat34 relativePose, 
	const unsigned int pointsNum, 
	const unsigned int offset,
	const int i
)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < pointsNum) {
		// 转换point到preAlignedSurfel中
		const float3 convertPoint = relativePose.rot * depthSurfel[idx].VertexAndConfidence + relativePose.trans;
		mergedSurfels[offset + idx].VertexAndConfidence = make_float4(convertPoint.x, convertPoint.y, convertPoint.z, depthSurfel[idx].VertexAndConfidence.w);
		const float3 convertNormal = relativePose.rot * depthSurfel[idx].NormalAndRadius;
		mergedSurfels[offset + idx].NormalAndRadius = make_float4(convertNormal.x, convertNormal.y, convertNormal.z, depthSurfel[idx].NormalAndRadius.w);
		mergedSurfels[offset + idx].pixelCoordinate = depthSurfel[idx].pixelCoordinate;
		mergedSurfels[offset + idx].ColorAndTime = depthSurfel[idx].ColorAndTime;
		//mergedSurfels[offset + idx].flag = i;
		//printf("mergedSurfels.flag =%u \n", mergedSurfels[offset + idx].flag);
		//if ((idx<10)&&(i==0))
		//{
		//	printf("mergedSurfels[offset + idx].NormalAndRadius= %f %f %f %f\n", mergedSurfels[offset + idx].NormalAndRadius.x, mergedSurfels[offset + idx].NormalAndRadius.y, mergedSurfels[offset + idx].NormalAndRadius.z, mergedSurfels[offset + idx].NormalAndRadius.w);
		//}
	}
}

void SparseSurfelFusion::SurfelsProcessor::MergeDenseSurfelToCanonicalField(DeviceBufferArray<DepthSurfel>& mergedSurfels, DeviceArrayView<DepthSurfel>& currentValidDepthSurfel, mat34 cameraPose, const unsigned int CameraID, const unsigned int offset, cudaStream_t stream)
{
	DeviceArrayHandle<DepthSurfel> mergePoints = mergedSurfels.ArrayHandle();		// 暴露指针
	DeviceArrayView<DepthSurfel> cameraPoints = currentValidDepthSurfel;			// 稠密面元
	size_t cameraPointsNum = currentValidDepthSurfel.Size();

	dim3 block(256);
	dim3 grid(divUp(cameraPointsNum, block.x));
	device::MergeDenseSurfelToCanonicalFieldKernel << <grid, block, 0, stream >> > (mergePoints, cameraPoints, cameraPose, cameraPointsNum, offset, CameraID);
}
