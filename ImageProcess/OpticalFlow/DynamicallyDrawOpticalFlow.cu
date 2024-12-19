#include "DynamicallyDrawOpticalFlow.h"
__global__ void SparseSurfelFusion::device::adjustModelPositionKernel(const mat34 AdjustMatrix, const unsigned int PointNum, float3* point, ColorVertex* vertex)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= PointNum) return;
	point[idx] = AdjustMatrix.rot * point[idx] + AdjustMatrix.trans;
	vertex[idx].coor = AdjustMatrix.rot * vertex[idx].coor + AdjustMatrix.trans;
}



__global__ void SparseSurfelFusion::device::adjustPointsCoordinate(const unsigned int pointsCount, float3* points, ColorVertex* vertex)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsCount) return;
	points[idx].x = (points[idx].x + 2.0f) * 2.0f / 3.0f - 1.0f;
	points[idx].y = (points[idx].y + 1.5f) * 2.0f / 3.0f - 1.0f;
	points[idx].z = (points[idx].z + 2.0f) * 2.0f / 3.0f - 1.0f;
	vertex[idx].coor.x = (vertex[idx].coor.x + 2.0f) * 2.0f / 3.0f - 1.0f;
	vertex[idx].coor.y = (vertex[idx].coor.y + 1.5f) * 2.0f / 3.0f - 1.0f;
	vertex[idx].coor.z = (vertex[idx].coor.z + 2.0f) * 2.0f / 3.0f - 1.0f;
}


void SparseSurfelFusion::DynamicallyDrawOpticalFlow::adjustPointsCoordinate(float3* points, ColorVertex* vertex, cudaStream_t stream)
{

	dim3 block(128);
	dim3 grid(divUp(ValidNum, block.x));
	device::adjustPointsCoordinate << <grid, block, 0, stream >> > (ValidNum, points, vertex);
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::adjustModelPosition(float3* points, ColorVertex* vertex, cudaStream_t stream)
{
	dim3 block(128);
	dim3 grid(divUp(ValidNum, block.x));
	device::adjustModelPositionKernel << <grid, block, 0, stream >> > (AdjustModelSE3, ValidNum, points, vertex);
}