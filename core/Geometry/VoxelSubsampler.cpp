/*****************************************************************//**
 * \file   VoxelSubsampler.h
 * \brief  用作对稠密顶点进行下采样
 *
 * \author LUO
 * \date   February 26th 2024
 *********************************************************************/
#include "VoxelSubsampler.h"

void SparseSurfelFusion::VoxelSubsampler::AllocateBuffer(unsigned int maxInputPoints)
{
	pointKey.AllocateBuffer(maxInputPoints);
	pointKeySort.AllocateBuffer(maxInputPoints);
	colorViewTimeSort.AllocateBuffer(maxInputPoints);
	voxelLabel.AllocateBuffer(maxInputPoints);
	voxelLabelPrefixsum.AllocateBuffer(maxInputPoints);
	// 压缩后的点数量不超过输入点的1/5
	const unsigned int compacted_max_size = maxInputPoints / Constants::maxCompactedInputPointsScale;		
	compactedVoxelKey.AllocateBuffer(compacted_max_size);
	compactedVoxelOffset.AllocateBuffer(compacted_max_size);
	subsampledPoint.AllocateBuffer(compacted_max_size);
	allocatedBuffer = true;
}

void SparseSurfelFusion::VoxelSubsampler::ReleaseBuffer()
{
	//Constants::kMaxNumSurfels
	pointKey.ReleaseBuffer();
	voxelLabel.ReleaseBuffer();

	//smaller buffer
	compactedVoxelKey.ReleaseBuffer();
	compactedVoxelOffset.ReleaseBuffer();
	subsampledPoint.ReleaseBuffer();
	allocatedBuffer = false;
}

bool SparseSurfelFusion::VoxelSubsampler::isBufferEmpty()
{
	return allocatedBuffer;
}

SparseSurfelFusion::DeviceArrayView<float4> SparseSurfelFusion::VoxelSubsampler::PerformSubsample(const DeviceArrayView<float4>& points, const float voxelSize, cudaStream_t stream)
{
	buildVoxelKeyForPoints(points, voxelSize, stream);
	sortCompactVoxelKeys(points, stream);
	collectSubsampledPoint(subsampledPoint, voxelSize, stream);
	return subsampledPoint.ArrayView();
}

void SparseSurfelFusion::VoxelSubsampler::PerformSubsample(const DeviceArrayView<float4>& points, DeviceBufferArray<float4>& subsampledVertice, const float voxelSize, cudaStream_t stream)
{
	buildVoxelKeyForPoints(points, voxelSize, stream);
	sortCompactVoxelKeys(points, stream);
	collectSubsampledPoint(subsampledVertice, voxelSize, stream);
}

void SparseSurfelFusion::VoxelSubsampler::PerformSubsample(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& subsampledPoints, const float voxelSize, cudaStream_t stream)
{
	buildVoxelKeyForPoints(points, voxelSize, stream);		
	sortCompactVoxelKeys(points, colorViewTime, stream);						//排列体素其编码及points（仍有对应关系）
	collectSynchronizeSubsampledPoint(subsampledPoints, voxelSize, stream);		//降采样并且拷贝到CPU的内存中
}
