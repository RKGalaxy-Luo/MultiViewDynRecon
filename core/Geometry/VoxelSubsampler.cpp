/*****************************************************************//**
 * \file   VoxelSubsampler.h
 * \brief  �����Գ��ܶ�������²���
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
	// ѹ����ĵ�����������������1/5
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
	sortCompactVoxelKeys(points, colorViewTime, stream);						//������������뼰points�����ж�Ӧ��ϵ��
	collectSynchronizeSubsampledPoint(subsampledPoints, voxelSize, stream);		//���������ҿ�����CPU���ڴ���
}
