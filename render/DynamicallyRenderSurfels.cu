#include "DynamicallyRenderSurfels.h"

__global__ void SparseSurfelFusion::device::AdjustPointsCoordinateAndColorKernel(DeviceArrayView<DepthSurfel> rawSurfels, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedSurfels* renderedSurfels)
{

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsCount) return;
	renderedSurfels[idx].vertexConfidence.x = (renderedSurfels[idx].vertexConfidence.x - center.x) * 2.0f / maxEdge - 1.0f;
	renderedSurfels[idx].vertexConfidence.y = (renderedSurfels[idx].vertexConfidence.y - center.y) * 2.0f / maxEdge - 1.0f;
	renderedSurfels[idx].vertexConfidence.z = (renderedSurfels[idx].vertexConfidence.z - center.z) * 2.0f / maxEdge - 1.0f;
	renderedSurfels[idx].vertexConfidence.w = renderedSurfels[idx].vertexConfidence.w;

	renderedSurfels[idx].normalRadius = rawSurfels[idx].NormalAndRadius;

	uchar3 color;
	float_decode_rgb(rawSurfels[idx].ColorAndTime.x, color);
	renderedSurfels[idx].rgbTime.z = color.x / 255.0f;
	renderedSurfels[idx].rgbTime.y = color.y / 255.0f;
	renderedSurfels[idx].rgbTime.x = color.z / 255.0f;
	renderedSurfels[idx].rgbTime.w = rawSurfels[idx].ColorAndTime.z;
}

__global__ void SparseSurfelFusion::device::AdjustModelPositionKernel(DeviceArrayView<DepthSurfel> rawSurfels, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedSurfels* renderedSurfels)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= PointNum) return;
	float3 adjustCoordinate = AdjustMatrix.rot * rawSurfels[idx].VertexAndConfidence + AdjustMatrix.trans;
	renderedSurfels[idx].vertexConfidence = make_float4(adjustCoordinate.x, adjustCoordinate.y, adjustCoordinate.z, rawSurfels[idx].VertexAndConfidence.w);
}

void SparseSurfelFusion::DynamicallyRenderSurfels::AdjustSurfelsCoordinateAndColor(DeviceArrayView<DepthSurfel> surfels, const float3 center, const float maxEdge, cudaStream_t stream)
{
	const unsigned int SurfelsNum = surfels.Size();
	RenderedSurfels.ResizeArrayOrException(SurfelsNum);
	dim3 block(128);
	dim3 grid(divUp(SurfelsNum, block.x));
	device::AdjustPointsCoordinateAndColorKernel << <grid, block, 0, stream >> > (surfels, center, maxEdge, SurfelsNum, RenderedSurfels.Ptr());
}

void SparseSurfelFusion::DynamicallyRenderSurfels::AdjustModelPosition(DeviceArrayView<DepthSurfel> rawSurfels, cudaStream_t stream)
{
	const unsigned int SurfelsNum = rawSurfels.Size();
	RenderedSurfels.ResizeArrayOrException(SurfelsNum);
	dim3 block(128);
	dim3 grid(divUp(SurfelsNum, block.x));
	device::AdjustModelPositionKernel << <grid, block, 0, stream >> > (rawSurfels, AdjustModelSE3, SurfelsNum, RenderedSurfels.Ptr());
}
